"""
Log-space diagnostics for the forecasting model's predictions.

Raw-pound R^2/MAE are dominated by a handful of extreme releases
(max ~400M lb vs. median ~95 lb), so this script evaluates error in
log space, which is the fairer view of model quality, and produces a
breakdown by quantity range for the report/presentation.

Run AFTER src/forecasting/train_final.py (or train.py), which writes:
    data/processed/forecasting_predictions_final.csv
    (columns: actual_quantity, actual_release_binary,
              release_probability, positive_quantity_prediction,
              expected_quantity)
"""

import json
import numpy as np
import pandas as pd

PREDICTIONS = "data/processed/forecasting_predictions_final.csv"
RESULTS_OUT = "results/forecasting/log_diagnostics.json"

df = pd.read_csv(PREDICTIONS)

actual = df["actual_quantity"]
prediction = df["expected_quantity"]

print("=" * 70)
print("FORECASTING DIAGNOSTICS")
print("=" * 70)

print("\nActual quantity statistics:")
print(actual.describe().to_string())

print("\nActual quantity percentiles:")
percentiles = [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
actual_percentiles = {}
for p in percentiles:
    value = float(np.percentile(actual, p))
    actual_percentiles[str(p)] = value
    print(f"{p:6.1f}% : {value:,.2f}")

print("\nPrediction statistics:")
print(prediction.describe().to_string())

print("\nPrediction percentiles:")
prediction_percentiles = {}
for p in percentiles:
    value = float(np.percentile(prediction, p))
    prediction_percentiles[str(p)] = value
    print(f"{p:6.1f}% : {value:,.2f}")

# ------------------------------------------------------------
# ERROR ANALYSIS
# ------------------------------------------------------------

error = prediction - actual
absolute_error = np.abs(error)

print("\nError statistics:")
print(absolute_error.describe().to_string())

# ------------------------------------------------------------
# ZERO VS POSITIVE
# ------------------------------------------------------------

zero_mask = actual == 0
positive_mask = actual > 0

print("\nZero-release records:")
print(f"Count: {int(zero_mask.sum()):,}")
print(f"Mean prediction: {prediction[zero_mask].mean():,.2f}")

print("\nPositive-release records:")
print(f"Count: {int(positive_mask.sum()):,}")
print(f"Mean actual:     {actual[positive_mask].mean():,.2f}")
print(f"Mean prediction: {prediction[positive_mask].mean():,.2f}")

# ------------------------------------------------------------
# EXTREME VALUES
# ------------------------------------------------------------

print("\nLargest actual releases:")

largest = pd.DataFrame({
    "actual": actual,
    "prediction": prediction,
    "absolute_error": absolute_error,
}).sort_values("actual", ascending=False).head(20)

print(largest.to_string(index=False))

# ------------------------------------------------------------
# LOG-SCALE DIAGNOSTICS
# ------------------------------------------------------------

actual_log = np.log1p(actual)
prediction_log = np.log1p(np.maximum(prediction, 0))

log_error = np.abs(actual_log - prediction_log)

print("\nLog-scale error:")
print(log_error.describe().to_string())

print("\nLog-scale error percentiles:")
log_error_percentiles = {}
for p in [50, 75, 90, 95, 99]:
    value = float(np.percentile(log_error, p))
    log_error_percentiles[str(p)] = value
    print(f"{p}% : {value:.4f}")

# ------------------------------------------------------------
# ERROR BY QUANTITY RANGE (useful "where does it fail" table)
# ------------------------------------------------------------

bins = [-1, 0, 100, 10_000, 1_000_000, np.inf]
labels = ["ZERO", "1-100 lb", "100-10k lb", "10k-1M lb", "1M+ lb"]

df["range"] = pd.cut(actual, bins=bins, labels=labels)

range_summary = (
    df.assign(absolute_error=absolute_error, log_error=log_error)
    .groupby("range", observed=True)
    .agg(
        count=("actual_quantity", "size"),
        mean_actual=("actual_quantity", "mean"),
        mean_prediction=("expected_quantity", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        mean_log_error=("log_error", "mean"),
    )
)

print("\nError by quantity range:")
print(range_summary.to_string())

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

import os
os.makedirs("results/forecasting", exist_ok=True)

diagnostics = {
    "actual_percentiles": actual_percentiles,
    "prediction_percentiles": prediction_percentiles,
    "mean_absolute_error": float(absolute_error.mean()),
    "median_absolute_error": float(absolute_error.median()),
    "zero_release_count": int(zero_mask.sum()),
    "zero_release_mean_prediction": float(prediction[zero_mask].mean()),
    "positive_release_count": int(positive_mask.sum()),
    "positive_release_mean_actual": float(actual[positive_mask].mean()),
    "positive_release_mean_prediction": float(prediction[positive_mask].mean()),
    "log_error_percentiles": log_error_percentiles,
    "error_by_quantity_range": json.loads(
        range_summary.reset_index().to_json(orient="records")
    ),
}

with open(RESULTS_OUT, "w") as f:
    json.dump(diagnostics, f, indent=2)

print(f"\nSaved: {RESULTS_OUT}")