"""
Compare final two-stage waste forecasting models.

Person A - Waste Generation Forecasting

Stage 1:
    Random Forest Classifier
    Predicts P(waste release > 0)

Stage 2:
    Compare:
        1. Random Forest Regressor
        2. HistGradientBoostingRegressor

Final forecast:

    P(release > 0)
    *
    predicted positive-release quantity
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading forecasting data...")

X_train = pd.read_csv(
    "data/processed/forecasting_X_train.csv"
)

X_test = pd.read_csv(
    "data/processed/forecasting_X_test.csv"
)

y_train = pd.read_csv(
    "data/processed/forecasting_y_train.csv"
)

y_test = pd.read_csv(
    "data/processed/forecasting_y_test.csv"
)


print(f"Training shape: {X_train.shape}")
print(f"Testing shape:  {X_test.shape}")


# ============================================================
# TARGETS
# ============================================================

train_positive = (
    y_train["release_binary"] == 1
)

test_positive = (
    y_test["release_binary"] == 1
)

y_train_log = y_train["log_release"]

y_test_actual = (
    y_test["total_release_current"]
)

y_test_binary = (
    y_test["release_binary"]
)


# ============================================================
# STAGE 1
# ============================================================

print()
print("=" * 70)
print("STAGE 1: RELEASE OCCURRENCE")
print("=" * 70)

# We only need the already-trained classifier
# for this comparison.

import joblib

classifier = joblib.load(
    "models/forecasting/release_classifier.joblib"
)

release_probability = classifier.predict_proba(
    X_test
)[:, 1]

print("Release probabilities loaded.")


# ============================================================
# POSITIVE-RELEASE TRAINING DATA
# ============================================================

X_train_positive = X_train.loc[
    train_positive
]

y_train_positive_log = y_train_log.loc[
    train_positive
]

print()
print(
    f"Positive training samples: "
    f"{len(X_train_positive):,}"
)


# ============================================================
# MODEL 1 — RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("MODEL 1: RANDOM FOREST REGRESSOR")
print("=" * 70)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

print("Training Random Forest...")

rf.fit(
    X_train_positive,
    y_train_positive_log,
)

rf_positive_quantity = np.expm1(
    rf.predict(X_test)
)

rf_positive_quantity = np.maximum(
    rf_positive_quantity,
    0
)

rf_expected = (
    release_probability
    * rf_positive_quantity
)


# ============================================================
# MODEL 2 — HISTOGRAM GRADIENT BOOSTING
# ============================================================

print()
print("=" * 70)
print("MODEL 2: HISTOGRAM GRADIENT BOOSTING")
print("=" * 70)

hgb = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=0.0,
    random_state=42,
)

print("Training HGB...")

hgb.fit(
    X_train_positive,
    y_train_positive_log,
)

hgb_positive_quantity = np.expm1(
    hgb.predict(X_test)
)

hgb_positive_quantity = np.maximum(
    hgb_positive_quantity,
    0
)

hgb_expected = (
    release_probability
    * hgb_positive_quantity
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    prediction
):

    mae = mean_absolute_error(
        y_test_actual,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_actual,
            prediction
        )
    )

    r2 = r2_score(
        y_test_actual,
        prediction
    )

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# FINAL COMPARISON
# ============================================================

results = []

results.append(
    evaluate_model(
        "Random Forest",
        rf_expected
    )
)

results.append(
    evaluate_model(
        "HistGradientBoosting",
        hgb_expected
    )
)


results_df = pd.DataFrame(results)


print()
print("=" * 70)
print("FINAL TWO-STAGE FORECAST COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE COMPARISON
# ============================================================

results_df.to_csv(
    "data/processed/"
    "forecasting_model_comparison.csv",
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_comparison = pd.DataFrame({

    "actual_quantity":
        y_test_actual.values,

    "release_probability":
        release_probability,

    "rf_positive_quantity":
        rf_positive_quantity,

    "rf_expected_quantity":
        rf_expected,

    "hgb_positive_quantity":
        hgb_positive_quantity,

    "hgb_expected_quantity":
        hgb_expected,

})

prediction_comparison.to_csv(
    "data/processed/"
    "forecasting_model_predictions.csv",
    index=False
)


print()
print("Saved:")
print(
    "data/processed/"
    "forecasting_model_comparison.csv"
)

print(
    "data/processed/"
    "forecasting_model_predictions.csv"
)