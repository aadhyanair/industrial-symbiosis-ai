import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

X_TRAIN = "data/processed/forecasting_X_train.csv"
X_TEST = "data/processed/forecasting_X_test.csv"

Y_TRAIN = "data/processed/forecasting_y_train.csv"
Y_TEST = "data/processed/forecasting_y_test.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

X_train = pd.read_csv(X_TRAIN)
X_test = pd.read_csv(X_TEST)

y_train = pd.read_csv(Y_TRAIN)
y_test = pd.read_csv(Y_TEST)


# Positive releases only
train_mask = y_train["release_binary"] == 1
test_mask = y_test["release_binary"] == 1

X_train_positive = X_train.loc[train_mask]
X_test_positive = X_test.loc[test_mask]

y_train_actual = y_train.loc[
    train_mask,
    "total_release_current"
]

y_test_actual = y_test.loc[
    test_mask,
    "total_release_current"
]


print("Training samples:", len(X_train_positive))
print("Testing samples:", len(X_test_positive))


# ============================================================
# MODEL CONFIG
# ============================================================

def make_model():

    return HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, y_train_transformed, inverse_transform):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    model = make_model()

    print("Training...")

    model.fit(
        X_train_positive,
        y_train_transformed
    )

    pred_transformed = model.predict(
        X_test_positive
    )

    pred_quantity = inverse_transform(
        pred_transformed
    )

    # Protect against numerical negatives
    pred_quantity = np.maximum(
        pred_quantity,
        0
    )

    actual_log = np.log1p(
        y_test_actual
    )

    pred_log = np.log1p(
        pred_quantity
    )

    # Raw metrics
    raw_mae = mean_absolute_error(
        y_test_actual,
        pred_quantity
    )

    raw_rmse = np.sqrt(
        mean_squared_error(
            y_test_actual,
            pred_quantity
        )
    )

    raw_r2 = r2_score(
        y_test_actual,
        pred_quantity
    )

    # Log metrics
    log_mae = mean_absolute_error(
        actual_log,
        pred_log
    )

    log_rmse = np.sqrt(
        mean_squared_error(
            actual_log,
            pred_log
        )
    )

    log_r2 = r2_score(
        actual_log,
        pred_log
    )

    print()
    print(f"Raw MAE:   {raw_mae:,.2f}")
    print(f"Raw RMSE:  {raw_rmse:,.2f}")
    print(f"Raw R2:    {raw_r2:.4f}")

    print()

    print(f"Log MAE:   {log_mae:.4f}")
    print(f"Log RMSE:  {log_rmse:.4f}")
    print(f"Log R2:    {log_r2:.4f}")

    return {
        "model": name,
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "raw_r2": raw_r2,
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "log_r2": log_r2
    }


# ============================================================
# EXPERIMENT 1 — LOG1P
# ============================================================

results = []

log_target = np.log1p(
    y_train_actual
)

results.append(
    evaluate(
        "LOG1P TARGET",
        log_target,
        lambda x: np.expm1(x)
    )
)


# ============================================================
# EXPERIMENT 2 — SQUARE ROOT
# ============================================================

sqrt_target = np.sqrt(
    y_train_actual
)

results.append(
    evaluate(
        "SQRT TARGET",
        sqrt_target,
        lambda x: np.square(x)
    )
)


# ============================================================
# EXPERIMENT 3 — FOURTH ROOT
# ============================================================

root4_target = np.power(
    y_train_actual,
    0.25
)

results.append(
    evaluate(
        "FOURTH ROOT TARGET",
        root4_target,
        lambda x: np.power(x, 4)
    )
)


# ============================================================
# FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

print()
print("=" * 60)
print("FINAL TARGET TRANSFORMATION COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)