import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# LOAD VALIDATION PREDICTIONS
# ============================================================

data = np.load(
    "data/processed/validation_predictions.npz"
)

y_val = data["y_val"]

rf = data["random_forest"]
xgb = data["xgboost"]


# ============================================================
# LOAD OPTIMIZED WEIGHTS
# ============================================================

weights = np.load(
    "data/processed/ensemble_weights.npy"
)

w_rf = weights[1]
w_xgb = weights[2]

print("Ensemble weights:")
print("Random Forest:", w_rf)
print("XGBoost:", w_xgb)


# ============================================================
# ENSEMBLE PROBABILITIES
# ============================================================

ensemble_probabilities = (
    w_rf * rf +
    w_xgb * xgb
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = np.argmax(
    ensemble_probabilities,
    axis=1
)


classes = [
    "energy_recovery",
    "no_management",
    "recycling",
    "treatment"
]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_val,
    y_pred
)


print("\n" + "=" * 70)
print("ENSEMBLE VALIDATION RESULTS")
print("=" * 70)

print(
    f"Accuracy:          {accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=classes,
        digits=4
    )
)


print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_val,
    y_pred
)

print(
    pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )
)


# ============================================================
# SAVE VALIDATION ENSEMBLE PREDICTIONS
# ============================================================

np.savez(
    "data/processed/ensemble_validation_results.npz",
    y_true=y_val,
    y_pred=y_pred,
    probabilities=ensemble_probabilities
)

print(
    "\nSaved:"
)

print(
    "data/processed/ensemble_validation_results.npz"
)