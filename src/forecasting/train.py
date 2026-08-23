"""
Two-stage waste generation forecasting.

Stage 1:
    Random Forest Classifier
    Predicts whether waste release > 0.

Stage 2:
    Random Forest Regressor
    Predicts log1p(waste quantity) for positive releases.

Final expected forecast:
    P(positive release) * predicted positive quantity
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

X_TRAIN = "data/processed/forecasting_X_train.csv"
X_TEST = "data/processed/forecasting_X_test.csv"

Y_TRAIN = "data/processed/forecasting_y_train.csv"
Y_TEST = "data/processed/forecasting_y_test.csv"

CLASSIFIER_OUT = "models/forecasting/release_classifier.joblib"
REGRESSOR_OUT = "models/forecasting/quantity_regressor.joblib"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading preprocessed data...")

X_train = pd.read_csv(X_TRAIN)
X_test = pd.read_csv(X_TEST)

y_train = pd.read_csv(Y_TRAIN)
y_test = pd.read_csv(Y_TEST)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape:  {X_test.shape}")


# ============================================================
# TARGETS
# ============================================================

y_train_binary = y_train["release_binary"]
y_test_binary = y_test["release_binary"]

y_train_log = y_train["log_release"]
y_test_log = y_test["log_release"]

y_train_actual = y_train["total_release_current"]
y_test_actual = y_test["total_release_current"]


# ============================================================
# STAGE 1 — RELEASE OCCURRENCE
# ============================================================

print()
print("=" * 60)
print("STAGE 1: POSITIVE RELEASE CLASSIFICATION")
print("=" * 60)

classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("Training Random Forest Classifier...")

classifier.fit(
    X_train,
    y_train_binary,
)


# ============================================================
# CLASSIFICATION PREDICTIONS
# ============================================================

release_probability = classifier.predict_proba(
    X_test
)[:, 1]

release_prediction = (
    release_probability >= 0.5
).astype(int)


accuracy = accuracy_score(
    y_test_binary,
    release_prediction,
)

roc_auc = roc_auc_score(
    y_test_binary,
    release_probability,
)


print()
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

print()
print("Classification Report:")

print(
    classification_report(
        y_test_binary,
        release_prediction,
        target_names=[
            "ZERO_RELEASE",
            "POSITIVE_RELEASE",
        ],
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test_binary,
        release_prediction,
    )
)


# ============================================================
# STAGE 2 — QUANTITY REGRESSION
# ============================================================

print()
print("=" * 60)
print("STAGE 2: POSITIVE RELEASE QUANTITY")
print("=" * 60)


positive_train_mask = (
    y_train_binary == 1
)

positive_test_mask = (
    y_test_binary == 1
)


X_train_positive = X_train.loc[
    positive_train_mask
]

y_train_positive_log = y_train_log.loc[
    positive_train_mask
]


X_test_positive = X_test.loc[
    positive_test_mask
]

y_test_positive_actual = y_test_actual.loc[
    positive_test_mask
]


print(
    f"Positive training samples: "
    f"{len(X_train_positive):,}"
)

print(
    f"Positive testing samples: "
    f"{len(X_test_positive):,}"
)


regressor = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


print()
print("Training Random Forest Regressor...")

regressor.fit(
    X_train_positive,
    y_train_positive_log,
)


# ============================================================
# QUANTITY PREDICTION
# ============================================================

predicted_log_quantity = regressor.predict(
    X_test_positive
)

predicted_quantity = np.expm1(
    predicted_log_quantity
)


# ============================================================
# REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_test_positive_actual,
    predicted_quantity,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_positive_actual,
        predicted_quantity,
    )
)

r2 = r2_score(
    y_test_positive_actual,
    predicted_quantity,
)

log_mae = mean_absolute_error(
    np.log1p(y_test_positive_actual),
    predicted_log_quantity,
)


print()
print("Positive-release quantity performance:")

print(
    f"MAE:      {mae:,.2f} pounds"
)

print(
    f"RMSE:     {rmse:,.2f} pounds"
)

print(
    f"R² Score: {r2:.4f}"
)

print(
    f"Log MAE:  {log_mae:.4f}"
)


# ============================================================
# FINAL EXPECTED FORECAST
# ============================================================
#
# Expected quantity =
#
# P(release > 0)
#       ×
# predicted quantity given release > 0
#
# For every test observation.
# ============================================================

positive_quantity_all = np.expm1(
    regressor.predict(X_test)
)

expected_forecast = (
    release_probability
    * positive_quantity_all
)


# ============================================================
# FINAL FORECAST METRICS
# ============================================================

final_mae = mean_absolute_error(
    y_test_actual,
    expected_forecast,
)

final_rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        expected_forecast,
    )
)

final_r2 = r2_score(
    y_test_actual,
    expected_forecast,
)


print()
print("=" * 60)
print("FINAL EXPECTED-QUANTITY FORECAST")
print("=" * 60)

print(
    f"MAE:      {final_mae:,.2f} pounds"
)

print(
    f"RMSE:     {final_rmse:,.2f} pounds"
)

print(
    f"R² Score: {final_r2:.4f}"
)


# ============================================================
# SAVE MODELS
# ============================================================

os.makedirs(
    "models/forecasting",
    exist_ok=True,
)


joblib.dump(
    classifier,
    CLASSIFIER_OUT,
)

joblib.dump(
    regressor,
    REGRESSOR_OUT,
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = pd.DataFrame(
    {
        "actual_quantity": y_test_actual.values,
        "actual_release_binary": y_test_binary.values,
        "release_probability": release_probability,
        "positive_quantity_prediction": positive_quantity_all,
        "expected_quantity": expected_forecast,
    }
)


prediction_path = (
    "data/processed/"
    "forecasting_predictions.csv"
)

predictions.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("Models saved:")
print(CLASSIFIER_OUT)
print(REGRESSOR_OUT)

print()
print("Predictions saved:")
print(prediction_path)