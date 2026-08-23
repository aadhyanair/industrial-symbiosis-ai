import pandas as pd
import numpy as np

PREDICTIONS = (
    "data/processed/forecasting_model_predictions.csv"
)

df = pd.read_csv(PREDICTIONS)

actual = df["actual_quantity"]
prediction = df["rf_expected_quantity"]

print("=" * 70)
print("FORECASTING DIAGNOSTICS")
print("=" * 70)

print("\nActual quantity statistics:")
print(actual.describe().to_string())

print("\nActual quantity percentiles:")

for p in [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
    print(
        f"{p:6.1f}% : "
        f"{np.percentile(actual, p):,.2f}"
    )

print("\nPrediction statistics:")
print(prediction.describe().to_string())

print("\nPrediction percentiles:")

for p in [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
    print(
        f"{p:6.1f}% : "
        f"{np.percentile(prediction, p):,.2f}"
    )

# ------------------------------------------------------------
# ERROR ANALYSIS
# ------------------------------------------------------------

error = prediction - actual
absolute_error = np.abs(error)

print("\nError statistics:")
print(
    absolute_error.describe().to_string()
)

# ------------------------------------------------------------
# ZERO VS POSITIVE
# ------------------------------------------------------------

zero_mask = actual == 0
positive_mask = actual > 0

print("\nZero-release records:")
print(f"Count: {zero_mask.sum():,}")
print(
    f"Mean prediction: "
    f"{prediction[zero_mask].mean():,.2f}"
)

print("\nPositive-release records:")
print(f"Count: {positive_mask.sum():,}")
print(
    f"Mean actual: "
    f"{actual[positive_mask].mean():,.2f}"
)
print(
    f"Mean prediction: "
    f"{prediction[positive_mask].mean():,.2f}"
)

# ------------------------------------------------------------
# EXTREME VALUES
# ------------------------------------------------------------

print("\nLargest actual releases:")

largest = pd.DataFrame({
    "actual": actual,
    "prediction": prediction,
    "absolute_error": absolute_error
}).sort_values(
    "actual",
    ascending=False
).head(20)

print(
    largest.to_string(index=False)
)

# ------------------------------------------------------------
# LOG-SCALE DIAGNOSTICS
# ------------------------------------------------------------

actual_log = np.log1p(actual)
prediction_log = np.log1p(
    np.maximum(prediction, 0)
)

log_error = np.abs(
    actual_log - prediction_log
)

print("\nLog-scale error:")
print(
    log_error.describe().to_string()
)

print("\nLog-scale percentiles:")

for p in [50, 75, 90, 95, 99]:
    print(
        f"{p}% : "
        f"{np.percentile(log_error, p):.4f}"
    )