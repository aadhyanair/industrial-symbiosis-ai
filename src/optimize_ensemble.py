import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from itertools import product


# ============================================================
# 1. LOAD VALIDATION PREDICTIONS
# ============================================================

print("Loading validation predictions...")

data = np.load(
    "data/processed/validation_predictions.npz"
)

y_val = data["y_val"]

lr = data["logistic_regression"]
rf = data["random_forest"]
xgb = data["xgboost"]
txgb = data["tuned_xgboost"]

print("Validation samples:", len(y_val))


# ============================================================
# 2. INDIVIDUAL MODEL PERFORMANCE
# ============================================================

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb,
    "Tuned XGBoost": txgb
}

print("\n" + "=" * 70)
print("INDIVIDUAL VALIDATION PERFORMANCE")
print("=" * 70)

for name, probabilities in models.items():

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    acc = accuracy_score(
        y_val,
        predictions
    )

    bal_acc = balanced_accuracy_score(
        y_val,
        predictions
    )

    print(
        f"{name:20s} "
        f"Accuracy={acc:.4f} "
        f"Balanced Accuracy={bal_acc:.4f}"
    )


# ============================================================
# 3. WEIGHT SEARCH
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING ENSEMBLE WEIGHTS")
print("=" * 70)

best_score = -1
best_accuracy = -1
best_weights = None


# weights are searched in increments of 0.1
weight_values = np.arange(
    0.0,
    1.01,
    0.1
)

total_combinations = 0


for w_lr, w_rf, w_xgb, w_txgb in product(
    weight_values,
    repeat=4
):

    # weights must sum to 1
    if abs(
        w_lr +
        w_rf +
        w_xgb +
        w_txgb - 1.0
    ) > 1e-9:
        continue

    total_combinations += 1

    ensemble_probabilities = (
        w_lr * lr +
        w_rf * rf +
        w_xgb * xgb +
        w_txgb * txgb
    )

    predictions = np.argmax(
        ensemble_probabilities,
        axis=1
    )

    score = balanced_accuracy_score(
        y_val,
        predictions
    )

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    if score > best_score:

        best_score = score
        best_accuracy = accuracy

        best_weights = (
            w_lr,
            w_rf,
            w_xgb,
            w_txgb
        )


print(
    "\nWeight combinations tested:",
    total_combinations
)

print("\nBEST ENSEMBLE")

print(
    f"Logistic Regression : {best_weights[0]:.1f}"
)

print(
    f"Random Forest       : {best_weights[1]:.1f}"
)

print(
    f"XGBoost             : {best_weights[2]:.1f}"
)

print(
    f"Tuned XGBoost       : {best_weights[3]:.1f}"
)

print(
    f"\nValidation Accuracy: "
    f"{best_accuracy:.4f}"
)

print(
    f"Validation Balanced Accuracy: "
    f"{best_score:.4f}"
)


# ============================================================
# 4. SAVE WEIGHTS
# ============================================================

np.save(
    "data/processed/ensemble_weights.npy",
    np.array(best_weights)
)

print(
    "\nSaved:"
)

print(
    "data/processed/ensemble_weights.npy"
)