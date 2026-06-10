import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from catboost import CatBoostRegressor
import nbformat as nbf
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def extract_time(df):
    df = df.copy()
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    return df

train = extract_time(train)
test = extract_time(test)
train = train.sort_values(['day', 'time_slot', 'Index']).reset_index(drop=True)

def base_transform(df):
    df = df.copy()
    df['is_morning_peak'] = df['hour'].between(9, 13).astype(int)
    df['is_night']        = df['hour'].between(0, 4).astype(int)
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_enc']     = (df['Landmarks'] == 'Yes').astype(int)
    
    road_map    = {'Residential': 0, 'Street': 1, 'Highway': 2}
    weather_map = {'Sunny': 0, 'Foggy': 1, 'Rainy': 2, 'Snowy': 3}
    df['RoadType_enc'] = df['RoadType'].map(road_map)
    df['Weather_enc']  = df['Weather'].map(weather_map)
    
    df['RoadType_enc'] = df.groupby('geohash')['RoadType_enc'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else -1)).fillna(-1)
    df['Temperature'] = df.groupby('geohash')['Temperature'].transform(lambda x: x.fillna(x.median())).fillna(df['Temperature'].median())
    df['Weather_enc'] = df['Weather_enc'].fillna(-1)
    
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather'] = df['Weather'].fillna('Unknown')
    df['LargeVehicles'] = df['LargeVehicles'].fillna('Unknown')
    df['Landmarks'] = df['Landmarks'].fillna('Unknown')
    
    return df

train = base_transform(train)
test = base_transform(test)

base_features = [
    'hour', 'minute', 'time_slot', 'is_morning_peak', 'is_night',
    'RoadType_enc', 'NumberofLanes', 'LargeVehicles_enc', 'Landmarks_enc',
    'Temperature', 'Weather_enc', 'day'
]

features = base_features + ['geo_mean_enc', 'geo_hour_mean_enc', 'geo_slot_enc']
cat_features = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
catboost_features = features + cat_features

def get_target_encodings_test(df_train, df_val, target_col='demand'):
    df_val_enc = df_val.copy()
    global_mean = df_train[target_col].mean()
    geo_map = df_train.groupby('geohash')[target_col].mean()
    df_val_enc['geo_mean_enc'] = df_val['geohash'].map(geo_map).fillna(global_mean)
    geo_hour_map = df_train.groupby(['geohash', 'hour'])[target_col].mean()
    df_val_enc['geo_hour_mean_enc'] = [geo_hour_map.get((g, h), global_mean) for g, h in zip(df_val['geohash'], df_val['hour'])]
    geo_slot_map = df_train.groupby(['geohash', 'time_slot'])[target_col].mean()
    df_val_enc['geo_slot_enc'] = [geo_slot_map.get((g, s), global_mean) for g, s in zip(df_val['geohash'], df_val['time_slot'])]
    return df_val_enc

def encode_group_expanding(X_tr):
    X_tr_enc = X_tr.copy()
    X_tr_enc['geo_sum'] = X_tr.groupby('geohash')['demand'].cumsum() - X_tr['demand']
    X_tr_enc['geo_count'] = X_tr.groupby('geohash').cumcount()
    global_mean = X_tr['demand'].mean()
    X_tr_enc['geo_mean_enc'] = np.where(X_tr_enc['geo_count'] > 0, X_tr_enc['geo_sum'] / X_tr_enc['geo_count'], global_mean)
    
    X_tr_enc['geo_hr_sum'] = X_tr.groupby(['geohash', 'hour'])['demand'].cumsum() - X_tr['demand']
    X_tr_enc['geo_hr_count'] = X_tr.groupby(['geohash', 'hour']).cumcount()
    X_tr_enc['geo_hour_mean_enc'] = np.where(X_tr_enc['geo_hr_count'] > 0, X_tr_enc['geo_hr_sum'] / X_tr_enc['geo_hr_count'], global_mean)
    
    X_tr_enc['geo_sl_sum'] = X_tr.groupby(['geohash', 'time_slot'])['demand'].cumsum() - X_tr['demand']
    X_tr_enc['geo_sl_count'] = X_tr.groupby(['geohash', 'time_slot']).cumcount()
    X_tr_enc['geo_slot_enc'] = np.where(X_tr_enc['geo_sl_count'] > 0, X_tr_enc['geo_sl_sum'] / X_tr_enc['geo_sl_count'], global_mean)
    return X_tr_enc

tscv = TimeSeriesSplit(n_splits=5)

lgb_preds = np.zeros(len(train))
cat_preds_all = {seed: np.zeros(len(train)) for seed in [42, 123, 2025, 777, 999]}

test_lgb_preds = np.zeros(len(test))
test_cat_preds_all = {seed: np.zeros(len(test)) for seed in [42, 123, 2025, 777, 999]}

lgb_scores = []

# Prepare Test target encodings using full train data
test_encoded = get_target_encodings_test(train, test, 'demand')
for col in ['geo_mean_enc', 'geo_hour_mean_enc', 'geo_slot_enc']:
    test[col] = test_encoded[col]

models_lgb = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
    X_tr, X_val = train.iloc[tr_idx].copy(), train.iloc[val_idx].copy()
    y_tr, y_val = train['demand'].iloc[tr_idx], train['demand'].iloc[val_idx]
    
    X_val = get_target_encodings_test(X_tr, X_val, 'demand')
    X_tr_enc = encode_group_expanding(X_tr)
    for col in ['geo_mean_enc', 'geo_hour_mean_enc', 'geo_slot_enc']:
        X_tr[col] = X_tr_enc[col]
        
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        objective='regression', learning_rate=0.03, n_estimators=600,
        num_leaves=127, min_child_samples=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    lgb_model.fit(X_tr[features], y_tr, eval_set=[(X_val[features], y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds_lgb = lgb_model.predict(X_val[features])
    lgb_preds[val_idx] = preds_lgb
    lgb_scores.append(max(0, 100 * r2_score(y_val, preds_lgb)))
    test_lgb_preds += lgb_model.predict(test[features]) / 5
    models_lgb.append(lgb_model)
    
    # CatBoost Models
    for seed in [42, 123, 2025, 777, 999]:
        cat_model = CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=8, l2_leaf_reg=3,
            cat_features=cat_features, random_seed=seed, verbose=0, task_type="CPU"
        )
        cat_model.fit(X_tr[catboost_features], y_tr, eval_set=(X_val[catboost_features], y_val), early_stopping_rounds=50)
        cat_preds_all[seed][val_idx] = cat_model.predict(X_val[catboost_features])
        test_cat_preds_all[seed] += cat_model.predict(test[catboost_features]) / 5

print(f"LightGBM Mean CV: {np.mean(lgb_scores):.4f}")

# Calculate CV for Blend
val_indices = np.concatenate([val_idx for _, val_idx in tscv.split(train)])
y_val_all = train['demand'].iloc[val_indices].values
lgb_p_all = lgb_preds[val_indices]

cat_avg_preds = np.mean([cat_preds_all[seed][val_indices] for seed in [42, 123, 2025, 777, 999]], axis=0)
cat_avg_cv = max(0, 100 * r2_score(y_val_all, cat_avg_preds))
print(f"CatBoost 5-Seed Avg CV: {cat_avg_cv:.4f}")

blend_preds = 0.30 * lgb_p_all + 0.70 * cat_avg_preds
blend_cv = max(0, 100 * r2_score(y_val_all, blend_preds))
print(f"LGBM (0.30) + CatBoost Avg (0.70) Blend CV: {blend_cv:.4f}")

# Extract top features
fi_df = pd.DataFrame({'feature': features, 'importance': models_lgb[-1].feature_importances_}).sort_values('importance', ascending=False)

# 5. Final Submission
test_cat_avg = np.mean([test_cat_preds_all[seed] for seed in [42, 123, 2025, 777, 999]], axis=0)
final_preds = 0.30 * test_lgb_preds + 0.70 * test_cat_avg
final_preds = np.clip(final_preds, 0, 1)

submission = pd.DataFrame({'Index': test['Index'], 'demand': final_preds})
submission.to_csv('submission.csv', index=False)
print("\nsubmission.csv generated!")

# Generate Notebook
md_intro_text = f"""# 🚦 Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction (Optimized)
**Evaluation metric:** `score = max(0, 100 * r2_score(actual, predicted))`

## Upgrades Applied
- **Validation**: Changed to `TimeSeriesSplit(n_splits=5)` to prevent temporal leakage into the past.
- **Target Encodings**: Fixed leakage using expanding mean over strictly past rows by timestamp.
- **Ensemble**: Evaluated LightGBM & 5-Seed CatBoost Averaging.
- **LightGBM Mean CV**: {np.mean(lgb_scores):.4f}
- **CatBoost 5-Seed Avg CV**: {cat_avg_cv:.4f}
- **Ensemble CV**: {blend_cv:.4f} using 0.30 LGBM + 0.70 Averaged CatBoost.
"""

with open('final_pipeline.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
import_start = lines.index("import pandas as pd\n")
generate_start = lines.index("# Generate Notebook\n")
code_full = "".join(lines[import_start:generate_start])

nb = nbf.v4.new_notebook()
md_intro = nbf.v4.new_markdown_cell(md_intro_text)
cell = nbf.v4.new_code_cell(code_full)
nb['cells'] = [md_intro, cell]

with open('gridlock_solution.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook updated!")
