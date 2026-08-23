import pandas as pd
import numpy as np
import joblib


print("Loading validation data...")

X_val = pd.read_csv(
    "data/processed/X_val.csv",
    low_memory=False
)

y_val = pd.read_csv(
    "data/processed/y_val.csv"
).squeeze()

print("Validation shape:", X_val.shape)


# ------------------------------------------------------------
# Load preprocessor
# ------------------------------------------------------------

print("\nLoading preprocessor...")

preprocessor = joblib.load(
    "data/processed/preprocessor.joblib"
)

X_val_processed = preprocessor.transform(X_val)

print(
    "Processed validation shape:",
    X_val_processed.shape
)


# ------------------------------------------------------------
# Target encoding
# ------------------------------------------------------------

classes = [
    "energy_recovery",
    "no_management",
    "recycling",
    "treatment"
]

label_to_int = {
    label: i
    for i, label in enumerate(classes)
}

y_val_encoded = y_val.map(label_to_int)


# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------

print("\nLoading models...")

models = {
    "logistic_regression":
        joblib.load(
            "data/processed/baseline_model.joblib"
        ),

    "random_forest":
        joblib.load(
            "data/processed/random_forest_model.joblib"
        ),

    "xgboost":
        joblib.load(
            "data/processed/xgboost_model.joblib"
        ),

    "tuned_xgboost":
        joblib.load(
            "data/processed/xgboost_validation_model.joblib"
        )
}


# ------------------------------------------------------------
# Generate probabilities
# ------------------------------------------------------------

predictions = {}

for name, model in models.items():

    print(
        f"\nGenerating probabilities: {name}"
    )

    probabilities = model.predict_proba(
        X_val_processed
    )

    predictions[name] = probabilities

    print(
        "Shape:",
        probabilities.shape
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

np.savez(
    "data/processed/validation_predictions.npz",
    y_val=y_val_encoded.to_numpy(),
    logistic_regression=predictions["logistic_regression"],
    random_forest=predictions["random_forest"],
    xgboost=predictions["xgboost"],
    tuned_xgboost=predictions["tuned_xgboost"]
)

print(
    "\nSaved:"
)

print(
    "data/processed/validation_predictions.npz"
)