# Gridlock Hackathon 2.0: Technical Walkthrough

## 1. Problem Understanding

**Objective**: Predict traffic demand for different geohashes across specific 15-minute time slots.
**Evaluation Metric**: The competition uses a bounded R² metric: `score = max(0, 100 * r2_score(actual, predicted))`. This rewards models that accurately predict variances in traffic volume and penalizes severe outliers, bounding the minimum score at 0.
**Dataset Characteristics**: The data contains chronological records containing categorical features (`RoadType`, `Weather`, `LargeVehicles`, `Landmarks`), spatial features (`geohash`, `NumberofLanes`), and temporal features (`day`, `timestamp`).

## 2. Exploratory Data Analysis

**Important Findings**:
- **Temporal Structure Discovery**: The training data spans Day 48 (all time slots 0-95) and the early part of Day 49 (time slots 0-8). Crucially, the test data spans the *future* of Day 49 (time slots 9-55). 
- **Missing Value Handling**: Certain columns like `RoadType` and `Temperature` had missing values. These were logically imputed using the most frequent (mode) and median values specific to their `geohash`, respectively.

## 3. Feature Engineering

### Retained Features
The following features consistently improved our strictly-evaluated validation score:
- `hour`, `minute`, `time_slot`: Extracted from the `timestamp` string.
- `is_morning_peak` & `is_night`: Binary flags highlighting critical daily traffic patterns.
- `day`: Retained to separate Day 48 from Day 49 trends.
- `RoadType_enc` & `Weather_enc`: Ordinal encodings mapping categorical severity to numerical scales.
- `geo_mean_enc`, `geo_hour_mean_enc`, `geo_slot_enc`: Extremely powerful historical target encodings that map the mean historical demand to the geohash, capturing spatial density.

### Rejected Features
During our incremental feature evaluation, the following features dropped our validation score and were excluded:
- **Cyclical time features** (`hour_sin`, `minute_cos`): These created unnecessary noise, as tree-based models natively handle step-based splits on the raw integer time_slots better.
- **Geohash frequency encoding**: Overfit the model to geohashes that appeared more frequently in the train set rather than generalizing to the test set.
- **Interaction features** (e.g., `RoadType` × `hour`): Caused curse-of-dimensionality overfitting within the decision trees.

## 4. Leakage Investigation

**The Problem with Random KFold**:
The original notebook used a 5-Fold random split. Because the test set exists entirely in the future, shuffling the data allowed future timestamps to leak into the validation folds of the past. Furthermore, the `target encoding` was calculated across KFold splits, meaning past rows were encoded using future demand data. This resulted in an artificially inflated CV score of ~99.23.

**The Fix**:
1. **TimeSeriesSplit**: We switched to `TimeSeriesSplit(n_splits=5)`, strictly validating on future blocks of data trained only on past blocks.
2. **Expanding Group Mean Target Encodings**: We replaced standard OOF encodings with an Expanding Group Mean. Each `time_slot` group is now encoded using the cumulative mean of the demand up to, but *excluding*, the current time_slot. This strictly prevents future-peeking.

## 5. Model Development

Three powerful Gradient Boosted Decision Tree (GBDT) algorithms were tuned:
- **LightGBM**: Tuned with `num_leaves=127`, `learning_rate=0.03`, and feature/bagging fractions of 0.8 to handle the tabular spatial-temporal data efficiently.
- **CatBoost**: Trained utilizing its powerful native categorical engine on `geohash`, `RoadType`, and `Weather`.
- **XGBoost**: Trained as a potential strong regularized contributor to the ensemble.
- **Hyperparameter Tuning**: All models utilized early stopping (`early_stopping_rounds=50`) against the TimeSeries validation sets to prevent overfitting.

## 6. Ensemble Strategy

We performed an extensive blending weight search grid across LightGBM, CatBoost, and XGBoost predictions.
- **Why CatBoost Performed Best**: CatBoost thrives on high-cardinality categoricals like `geohash`. Paired with the new un-leaked temporal target encodings, CatBoost dominated the standalone CV.
- **Multi-Seed CatBoost Averaging**: To stabilize CatBoost's predictions, we trained 5 separate models with different random seeds (42, 123, 2025, 777, 999) and averaged their predictions. This smoothing technique boosted CatBoost's CV dramatically.
- **Final Ensemble Weights**: The grid search settled on `0.30 * LightGBM + 0.70 * CatBoost (5-seed average)`. XGBoost was assigned a weight of `0.00` because it failed to provide any orthogonal predictive value over the other two models.

## 7. Results

Our final, rigorous, leakage-free cross-validation scores are as follows:

| Model | Leakage-Free CV |
|-------|-----------------|
| LightGBM | `84.45` |
| XGBoost | `82.22` |
| CatBoost (5-seed average) | `87.97` |
| **Final Ensemble (30% LGB / 70% CatBoost)** | **`88.06`** |

The final competition submission was generated using the ensemble:
0.30 × LightGBM + 0.70 × 5-seed averaged CatBoost.

## 8. Conclusion

**Lessons Learned**: Always analyze the temporal distribution of train and test sets before blindly applying random K-Fold CV. Target encodings are incredibly powerful, but they are the most common source of fatal leaderboard leakage if not applied chronologically.
**Strengths**: The final solution is mathematically robust against future leakage, uses a highly stabilized 5-seed averaged CatBoost backbone, and utilizes early stopping to perfectly adapt to chronological shifts.
**Future Improvements**: If more data were available, we could test recurrent neural networks (LSTMs) or spatial-temporal graph networks (STGCN) to capture the geographical adjacency of the geohashes over time.
