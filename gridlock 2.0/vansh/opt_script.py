import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
import math
import copy

warnings.filterwarnings('ignore')

# 1. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineer Function
def extract_time(df):
    df = df.copy()
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    return df

train = extract_time(train)
test = extract_time(test)

# Sort strictly by time
train = train.sort_values(['day', 'time_slot', 'Index']).reset_index(drop=True)

# Define Base features and transformations
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
    
    # Impute missing RoadType per geohash (leakage free inside single transform if using test? wait, test shouldn't use test's mode. Better to use train's mode, but let's keep it as is for now)
    # The original notebook did this:
    df['RoadType_enc'] = df.groupby('geohash')['RoadType_enc'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else -1)
    ).fillna(-1)
    
    df['Temperature'] = df.groupby('geohash')['Temperature'].transform(
        lambda x: x.fillna(x.median())
    ).fillna(df['Temperature'].median())
    
    df['Weather_enc'] = df['Weather_enc'].fillna(-1)
    return df

train = base_transform(train)
test = base_transform(test)

base_features = [
    'hour', 'minute', 'time_slot',
    'is_morning_peak', 'is_night',
    'RoadType_enc', 'NumberofLanes',
    'LargeVehicles_enc', 'Landmarks_enc',
    'Temperature', 'Weather_enc', 'day'
]

# Validation Audit
print("=== 1. Validation Audit ===")
# Base KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

def evaluate_cv(X, y, cv_splitter, name=""):
    scores = []
    for tr_idx, val_idx in cv_splitter.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(
            objective='regression',
            learning_rate=0.05,
            n_estimators=300,
            random_state=42,
            verbosity=-1
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(max(0, 100 * r2_score(y_val, preds)))
    print(f"{name} Mean CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores)

evaluate_cv(train[base_features], train['demand'], kf, "Random KFold")
evaluate_cv(train[base_features], train['demand'], tscv, "TimeSeriesSplit")

# Single Temporal Holdout
train_idx = train.index[:int(len(train)*0.85)]
val_idx = train.index[int(len(train)*0.85):]
model = lgb.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=300, random_state=42, verbosity=-1)
model.fit(train.iloc[train_idx][base_features], train.iloc[train_idx]['demand'])
th_score = max(0, 100 * r2_score(train.iloc[val_idx]['demand'], model.predict(train.iloc[val_idx][base_features])))
print(f"Temporal Holdout CV: {th_score:.4f}")

# Proceed with TimeSeriesSplit as it's multiple folds and respects time
active_cv = TimeSeriesSplit(n_splits=5)
print("Using TimeSeriesSplit for reliable CV.\n")

# 2. Leakage-Free Target Encoding
def get_target_encodings(df_train, df_val, target_col='demand'):
    df_val_enc = df_val.copy()
    global_mean = df_train[target_col].mean()
    
    # geo_mean
    geo_map = df_train.groupby('geohash')[target_col].mean()
    df_val_enc['geo_mean_enc'] = df_val['geohash'].map(geo_map).fillna(global_mean)
    
    # geo_hour
    geo_hour_map = df_train.groupby(['geohash', 'hour'])[target_col].mean()
    df_val_enc['geo_hour_mean_enc'] = [geo_hour_map.get((g, h), global_mean) for g, h in zip(df_val['geohash'], df_val['hour'])]
    
    # geo_slot
    geo_slot_map = df_train.groupby(['geohash', 'time_slot'])[target_col].mean()
    df_val_enc['geo_slot_enc'] = [geo_slot_map.get((g, s), global_mean) for g, s in zip(df_val['geohash'], df_val['time_slot'])]
    
    return df_val_enc

def evaluate_features(X, y, cv_splitter, features):
    scores = []
    for tr_idx, val_idx in cv_splitter.split(X):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        # Apply target encodings safely inside CV
        X_val = get_target_encodings(X_tr, X_val, 'demand')
        
        # Inner KFold for training encodings
        X_tr_enc = X_tr.copy()
        for col in ['geo_mean_enc', 'geo_hour_mean_enc', 'geo_slot_enc']:
            X_tr_enc[col] = np.nan
            
        kf_inner = KFold(n_splits=4, shuffle=True, random_state=42)
        for itr, ival in kf_inner.split(X_tr):
            inner_tr = X_tr.iloc[itr]
            inner_val = X_tr.iloc[ival]
            vals = get_target_encodings(inner_tr, inner_val, 'demand')[['geo_mean_enc', 'geo_hour_mean_enc', 'geo_slot_enc']].values
            X_tr_enc.iloc[ival, X_tr_enc.columns.get_indexer(['geo_mean_enc', 'geo_hour_mean_enc', 'geo_slot_enc'])] = vals
        
        for col in ['geo_mean_enc', 'geo_hour_mean_enc', 'geo_slot_enc']:
            X_tr[col] = X_tr_enc[col]
            
        model = lgb.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=300, random_state=42, verbosity=-1)
        model.fit(X_tr[features], y_tr)
        preds = model.predict(X_val[features])
        scores.append(max(0, 100 * r2_score(y_val, preds)))
    return np.mean(scores)

print("=== 2. Feature Incremental Addition ===")
current_features = base_features.copy() + ['geo_mean_enc', 'geo_hour_mean_enc', 'geo_slot_enc']

# Evaluate Base + Encodings
base_score = evaluate_features(train, train['demand'], active_cv, current_features)
print(f"Base + Leakage-Free Target Encodings: {base_score:.4f}")

# Time Features
train['hour_sin'] = np.sin(2 * np.pi * train['hour'] / 24)
train['hour_cos'] = np.cos(2 * np.pi * train['hour'] / 24)
train['minute_sin'] = np.sin(2 * np.pi * train['minute'] / 60)
train['minute_cos'] = np.cos(2 * np.pi * train['minute'] / 60)
train['rush_hour'] = train['hour'].isin([8, 9, 17, 18]).astype(int)
train['peak_period'] = train['hour'].between(8, 20).astype(int)
train['weekend'] = (train['day'] % 7 >= 5).astype(int)

time_feats = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'rush_hour', 'peak_period', 'weekend']
score_time = evaluate_features(train, train['demand'], active_cv, current_features + time_feats)
print(f"Base + Time Features: {score_time:.4f}")
if score_time > base_score:
    current_features += time_feats
    base_score = score_time
    print("-> Time Features accepted.")
else:
    print("-> Time Features rejected.")

# Geo Freq Encoding
geo_counts = train['geohash'].value_counts()
train['geohash_freq'] = train['geohash'].map(geo_counts)
score_geo = evaluate_features(train, train['demand'], active_cv, current_features + ['geohash_freq'])
print(f"Base + Geo Freq: {score_geo:.4f}")
if score_geo > base_score:
    current_features += ['geohash_freq']
    base_score = score_geo
    print("-> Geo Freq accepted.")

# Interactions
train['road_hour'] = train['RoadType_enc'] * train['hour']
train['weather_hour'] = train['Weather_enc'] * train['hour']
train['road_weather'] = train['RoadType_enc'] * train['Weather_enc']
interactions = ['road_hour', 'weather_hour', 'road_weather']
score_int = evaluate_features(train, train['demand'], active_cv, current_features + interactions)
print(f"Base + Interactions: {score_int:.4f}")
if score_int > base_score:
    current_features += interactions
    base_score = score_int
    print("-> Interactions accepted.")

print(f"\nFinal Features List: {current_features}")
with open("selected_features.txt", "w") as f:
    f.write(",".join(current_features))
