"""
Multi-year temporal forecasting training script.
Trains on 2021 data and evaluates on 2022 data using prior year lags.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT = "data/processed/tri_multi_year_panel.csv"

print("Loading multi-year panel dataset...")
df = pd.read_csv(INPUT, low_memory=False)

df["quantity_current_year"] = pd.to_numeric(
    df["quantity_current_year"], errors="coerce"
).fillna(0)
df["quantity_prior_year"] = pd.to_numeric(
    df["quantity_prior_year"], errors="coerce"
).fillna(0)

# Temporal Split: Train on 2021, Test on 2022
train_df = df[df["year"] == 2021].copy()
test_df = df[df["year"] == 2022].copy()

print(f"Training samples (2021): {len(train_df):,}")
print(f"Testing samples (2022):  {len(test_df):,}")

categorical_features = [
    "facility_state",
    "primary_naics_code",
    "cas_number",
    "chemical_name",
    "prod_ratio_or_activity",
]

numeric_features = [
    "latitude",
    "longitude",
    "prod_ratio_activity_index",
    "quantity_prior_year",  # True temporal lag feature
]

X_train = train_df[categorical_features + numeric_features].copy()
X_test = test_df[categorical_features + numeric_features].copy()

y_train_actual = train_df["quantity_current_year"]
y_test_actual = test_df["quantity_current_year"]

y_train_binary = (y_train_actual > 0).astype(int)
y_test_binary = (y_test_actual > 0).astype(int)

y_train_log = np.log1p(y_train_actual)
y_test_log = np.log1p(y_test_actual)

# Build Preprocessing Pipelines
print("\nBuilding preprocessing pipeline...")
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed feature dimensions: {X_train_processed.shape}")

# Stage 1: Release Occurrence Classification
print("\n" + "=" * 60)
print("STAGE 1: TEMPORAL RELEASE CLASSIFICATION (2021 -> 2022)")
print("=" * 60)

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

classifier.fit(X_train_processed, y_train_binary)

release_probability = classifier.predict_proba(X_test_processed)[:, 1]
release_prediction = (release_probability >= 0.5).astype(int)

accuracy = accuracy_score(y_test_binary, release_prediction)
roc_auc = roc_auc_score(y_test_binary, release_probability)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

# Stage 2: Positive Quantity Regression
print("\n" + "=" * 60)
print("STAGE 2: TEMPORAL QUANTITY REGRESSION (2021 -> 2022)")
print("=" * 60)

pos_train_mask = y_train_binary == 1
pos_test_mask = y_test_binary == 1

X_train_pos = X_train_processed[pos_train_mask]
y_train_pos_log = y_train_log.loc[pos_train_mask]

X_test_pos = X_test_processed[pos_test_mask]
y_test_pos_actual = y_test_actual.loc[pos_test_mask]

regressor = RandomForestRegressor(
    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
)

regressor.fit(X_train_pos, y_train_pos_log)

predicted_log_qty = regressor.predict(X_test_pos)
predicted_qty = np.expm1(predicted_log_qty)

mae = mean_absolute_error(y_test_pos_actual, predicted_qty)
rmse = np.sqrt(mean_squared_error(y_test_pos_actual, predicted_qty))
r2 = r2_score(y_test_pos_actual, predicted_qty)

print(f"Positive Quantity MAE:  {mae:,.2f} pounds")
print(f"Positive Quantity RMSE: {rmse:,.2f} pounds")
print(f"Positive Quantity R²:   {r2:.4f}")

# Save models
os.makedirs("models/forecasting_temporal", exist_ok=True)
joblib.dump(classifier, "models/forecasting_temporal/classifier.joblib")
joblib.dump(regressor, "models/forecasting_temporal/regressor.joblib")
joblib.dump(preprocessor, "models/forecasting_temporal/preprocessor.joblib")
print("\nMulti-year temporal models saved successfully!")