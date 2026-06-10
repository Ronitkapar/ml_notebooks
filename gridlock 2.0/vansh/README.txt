Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction
===========================================================

1. Problem Statement
--------------------
The objective of this competition is to predict traffic demand for various geohashes across different time slots. The evaluation metric for the competition is defined as `max(0, 100 * r2_score(actual, predicted))`, heavily rewarding models that can accurately capture variances in demand without overfitting.

2. Dataset Overview
-------------------
The dataset consists of traffic records spanning two days: Day 48 and Day 49.
- Train dataset: 77,299 rows containing data from Day 48 and early Day 49.
- Test dataset: 41,778 rows containing future data from Day 49 (time_slots 9-55).
The strictly chronological nature of the train/test split makes temporal awareness crucial to avoiding leaderboard shakeup.

3. Approach Summary
-------------------
Our solution leverages a rigorous, leakage-free TimeSeriesSplit cross-validation framework. We feature-engineered temporal and location-based target encodings, carefully applying an expanding group-mean strategy to prevent future data from leaking into past rows. The final predictions are a weighted ensemble of LightGBM and a 5-seed averaged CatBoost model.

4. Data Preprocessing & Feature Engineering
-----------------------------------------
Base processing included mapping categorical variables (e.g., RoadType, Weather) to numerical ordinals, and imputing missing RoadType and Temperature using geohash grouping. 

Retained Features:
- Time: hour, minute, time_slot, is_morning_peak, is_night
- Location/Categorical: RoadType_enc, NumberofLanes, LargeVehicles_enc, Landmarks_enc, Temperature, Weather_enc, day
- Target Encodings: geo_mean_enc, geo_hour_mean_enc, geo_slot_enc (Calculated using an Expanding Group Mean strictly on past historical rows)

Tested but Rejected Features (due to CV degradation):
- Cyclical time features (sin/cos of hour/minute)
- Geohash frequency encodings
- Interaction features (RoadType x Weather, etc.)

5. Validation Strategy
----------------------
- Removed random K-Fold CV to completely eliminate future-peeking.
- Implemented `TimeSeriesSplit(n_splits=5)` to simulate the future prediction task accurately.

6. Models & Ensemble Strategy
-----------------------------
We tuned three primary gradient boosting models:
- LightGBM (CV: 84.45)
- CatBoost (5-seed average) (CV: 87.97)
- XGBoost (CV: 82.22)

Ensemble Blend Search determined the optimal combination to be:
0.30 LightGBM
0.70 CatBoost (5-seed average)

Final Leakage-Free Cross Validation Score: 88.06

7. Tools & Libraries
--------------------
- Python 3
- Pandas, NumPy
- Scikit-Learn (TimeSeriesSplit, r2_score)
- LightGBM, CatBoost, XGBoost

8. Reproducibility Instructions
-------------------------------
To reproduce the final submission:
1. Ensure `train.csv` and `test.csv` are in the same directory as the scripts.
2. Run `python final_pipeline.py`.
3. The script will preprocess the data, generate target encodings, train the LightGBM model, train the 5 differently seeded CatBoost models, and output `submission.csv` to the working directory.

## Project Structure
gridlock_solution.ipynb   - Complete notebook with training and inference
final_pipeline.py         - Final production pipeline
opt_script.py             - Optimization experiments and model search
walkthrough.md            - Detailed technical explanation
selected_features.txt     - Final selected feature list
