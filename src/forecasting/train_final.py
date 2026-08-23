"""
Final production two-stage waste-generation forecasting model.

This replaces the Stage 2 model with the best-performing configuration
found during experimentation (experiment_hgb_tuned.py showed
HistGradientBoostingRegressor with more trees + deeper leaves gives
the best log-space R^2, which is the fair metric given the extreme
right-skew of the target).

Stage 1 (unchanged, already strong):
    RandomForestClassifier -> P(release > 0)

Stage 2 (upgraded):
    HistGradientBoostingRegressor -> log1p(quantity) | release > 0

Final expected forecast:
    P(release > 0) * expm1(predicted log quantity)

Saves:
    models/forecasting/release_classifier_final.joblib
    models/forecasting/quantity_regressor_final.joblib
    data/processed/forecasting_predictions_final.csv
    results/forecasting/forecast_results.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
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

CLASSIFIER_OUT = "models/forecasting/release_classifier_final.joblib"
REGRESSOR_OUT = "models/forecasting/quantity_regressor_final.joblib"

PREDICTIONS_OUT = "data/processed/forecasting_predictions_final.csv"
RESULTS_OUT = "results/forecasting/forecast_results.json"

RANDOM_STATE = 42

# Best config found in experiment_hgb_tuned.py (HGB_MORE_TREES:
# highest log_r2 = 0.4875)
REGRESSOR_PARAMS = dict(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=0.0,
    random_state=RANDOM_STATE,
)


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

y_train_actual = y_train["total_release_current"]
y_test_actual = y_test["total_release_current"]


# ============================================================
# STAGE 1 - RELEASE OCCURRENCE (unchanged, already strong)
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

classifier.fit(X_train, y_train_binary)

release_probability = classifier.predict_proba(X_test)[:, 1]
release_prediction = (release_probability >= 0.5).astype(int)

accuracy = accuracy_score(y_test_binary, release_prediction)
roc_auc = roc_auc_score(y_test_binary, release_probability)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

report_dict = classification_report(
    y_test_binary,
    release_prediction,
    target_names=["ZERO_RELEASE", "POSITIVE_RELEASE"],
    output_dict=True,
)

print("\nClassification Report:")
print(
    classification_report(
        y_test_binary,
        release_prediction,
        target_names=["ZERO_RELEASE", "POSITIVE_RELEASE"],
    )
)

conf_matrix = confusion_matrix(y_test_binary, release_prediction)

print("Confusion Matrix:")
print(conf_matrix)


# ============================================================
# STAGE 2 - QUANTITY REGRESSION (HistGradientBoosting, tuned)
# ============================================================

print()
print("=" * 60)
print("STAGE 2: POSITIVE RELEASE QUANTITY (HGB, tuned)")
print("=" * 60)

positive_train_mask = (y_train_binary == 1)
positive_test_mask = (y_test_binary == 1)

X_train_positive = X_train.loc[positive_train_mask]
y_train_positive_log = y_train_log.loc[positive_train_mask]

X_test_positive = X_test.loc[positive_test_mask]
y_test_positive_actual = y_test_actual.loc[positive_test_mask]

print(f"Positive training samples: {len(X_train_positive):,}")
print(f"Positive testing samples:  {len(X_test_positive):,}")

regressor = HistGradientBoostingRegressor(**REGRESSOR_PARAMS)

print("\nTraining HistGradientBoostingRegressor...")

regressor.fit(X_train_positive, y_train_positive_log)

predicted_log_quantity = regressor.predict(X_test_positive)
predicted_quantity = np.maximum(np.expm1(predicted_log_quantity), 0)

mae = mean_absolute_error(y_test_positive_actual, predicted_quantity)
rmse = np.sqrt(mean_squared_error(y_test_positive_actual, predicted_quantity))
r2 = r2_score(y_test_positive_actual, predicted_quantity)

log_mae = mean_absolute_error(
    np.log1p(y_test_positive_actual), predicted_log_quantity
)
log_rmse = np.sqrt(
    mean_squared_error(np.log1p(y_test_positive_actual), predicted_log_quantity)
)
log_r2 = r2_score(np.log1p(y_test_positive_actual), predicted_log_quantity)

print("\nPositive-release quantity performance (raw pounds):")
print(f"MAE:      {mae:,.2f}")
print(f"RMSE:     {rmse:,.2f}")
print(f"R2:       {r2:.4f}")

print("\nPositive-release quantity performance (log space - fairer metric):")
print(f"Log MAE:  {log_mae:.4f}")
print(f"Log RMSE: {log_rmse:.4f}")
print(f"Log R2:   {log_r2:.4f}")


# ============================================================
# FINAL EXPECTED FORECAST
# ============================================================

positive_quantity_all = np.maximum(
    np.expm1(regressor.predict(X_test)), 0
)

expected_forecast = release_probability * positive_quantity_all

final_mae = mean_absolute_error(y_test_actual, expected_forecast)
final_rmse = np.sqrt(mean_squared_error(y_test_actual, expected_forecast))
final_r2 = r2_score(y_test_actual, expected_forecast)

print()
print("=" * 60)
print("FINAL EXPECTED-QUANTITY FORECAST")
print("=" * 60)

print(f"MAE:  {final_mae:,.2f} pounds")
print(f"RMSE: {final_rmse:,.2f} pounds")
print(f"R2:   {final_r2:.4f}")


# ============================================================
# SAVE MODELS
# ============================================================

os.makedirs("models/forecasting", exist_ok=True)
os.makedirs("results/forecasting", exist_ok=True)

joblib.dump(classifier, CLASSIFIER_OUT)
joblib.dump(regressor, REGRESSOR_OUT)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "actual_quantity": y_test_actual.values,
    "actual_release_binary": y_test_binary.values,
    "release_probability": release_probability,
    "positive_quantity_prediction": positive_quantity_all,
    "expected_quantity": expected_forecast,
})

predictions.to_csv(PREDICTIONS_OUT, index=False)


# ============================================================
# SAVE RESULTS SUMMARY (for report / presentation)
# ============================================================

results_summary = {
    "stage_1_classification": {
        "model": "RandomForestClassifier",
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report_dict,
    },
    "stage_2_regression": {
        "model": "HistGradientBoostingRegressor",
        "params": REGRESSOR_PARAMS,
        "positive_train_samples": int(len(X_train_positive)),
        "positive_test_samples": int(len(X_test_positive)),
        "raw_mae": mae,
        "raw_rmse": rmse,
        "raw_r2": r2,
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "log_r2": log_r2,
    },
    "final_expected_forecast": {
        "mae": final_mae,
        "rmse": final_rmse,
        "r2": final_r2,
    },
    "notes": (
        "Raw-pound R2 is near zero because the target is extremely "
        "right-skewed (median ~95 lb, 99th pct ~450k lb, max ~400M lb). "
        "Log-space R2 (~0.48) is the fairer metric and shows the model "
        "captures meaningful signal across orders of magnitude. This is "
        "single-year TRI data, so the model predicts expected waste "
        "release given facility/material characteristics, not a true "
        "multi-year temporal forecast."
    ),
}

with open(RESULTS_OUT, "w") as f:
    json.dump(results_summary, f, indent=2)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("Models saved:")
print(CLASSIFIER_OUT)
print(REGRESSOR_OUT)

print()
print("Predictions saved:")
print(PREDICTIONS_OUT)

print()
print("Results summary saved:")
print(RESULTS_OUT)