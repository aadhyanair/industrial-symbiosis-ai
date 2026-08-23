import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading test data...")

X_test = pd.read_csv(
    "data/processed/X_test.csv",
    low_memory=False
)

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze()

print("Test shape:", X_test.shape)

print("\nLoading preprocessor...")
preprocessor = joblib.load(
    "data/processed/preprocessor.joblib"
)

X_test_processed = preprocessor.transform(X_test)

print("Processed test shape:", X_test_processed.shape)

print("\nLoading models...")

rf = joblib.load(
    "data/processed/random_forest_model.joblib"
)

xgb = joblib.load(
    "data/processed/xgboost_model.joblib"
)

# --------------------------------------------------
# IMPORTANT:
# Use the same class ordering used during training
# --------------------------------------------------

classes = [
    "energy_recovery",
    "no_management",
    "recycling",
    "treatment"
]

print("\nGenerating probabilities...")

rf_prob = rf.predict_proba(X_test_processed)
xgb_prob = xgb.predict_proba(X_test_processed)

print("RF probability shape:", rf_prob.shape)
print("XGBoost probability shape:", xgb_prob.shape)

# --------------------------------------------------
# Frozen ensemble weights from validation
# RF = 0.4
# XGBoost = 0.6
# --------------------------------------------------

rf_weight = 0.4
xgb_weight = 0.6

ensemble_prob = (
    rf_weight * rf_prob +
    xgb_weight * xgb_prob
)

pred_indices = np.argmax(ensemble_prob, axis=1)

predictions = np.array(classes)[pred_indices]

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print("FINAL TEST ENSEMBLE RESULTS")
print("=" * 60)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=classes,
        target_names=classes,
        digits=4
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=classes
)

print(
    pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )
)

# --------------------------------------------------
# Save final results
# --------------------------------------------------

np.savez(
    "data/processed/final_ensemble_results.npz",
    y_true=y_test.to_numpy(),
    predictions=predictions,
    probabilities=ensemble_prob
)

print("\nSaved:")
print("data/processed/final_ensemble_results.npz")