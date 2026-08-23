import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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

train_mask = y_train["release_binary"] == 1
test_mask = y_test["release_binary"] == 1

X_train = X_train.loc[train_mask]
X_test = X_test.loc[test_mask]

y_train_log = y_train.loc[
    train_mask, "log_release"
]

y_test_actual = y_test.loc[
    test_mask, "total_release_current"
]

y_test_log = np.log1p(y_test_actual)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


configs = [
    {
        "name": "HGB_BASELINE",
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
    },
    {
        "name": "HGB_MORE_TREES",
        "max_iter": 500,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
    },
    {
        "name": "HGB_DEEPER",
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "l2_regularization": 0.0,
    },
    {
        "name": "HGB_MORE_TREES_DEEPER",
        "max_iter": 500,
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "l2_regularization": 0.0,
    },
    {
        "name": "HGB_REGULARIZED",
        "max_iter": 500,
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "l2_regularization": 1.0,
    },
    {
        "name": "HGB_SLOW_LEARNING",
        "max_iter": 500,
        "learning_rate": 0.03,
        "max_leaf_nodes": 63,
        "l2_regularization": 1.0,
    },
]


results = []


for config in configs:

    print()
    print("=" * 60)
    print(config["name"])
    print("=" * 60)

    name = config.pop("name")

    model = HistGradientBoostingRegressor(
        max_iter=config["max_iter"],
        learning_rate=config["learning_rate"],
        max_leaf_nodes=config["max_leaf_nodes"],
        l2_regularization=config["l2_regularization"],
        random_state=42,
    )

    print("Training...")

    model.fit(
        X_train,
        y_train_log
    )

    pred_log = model.predict(X_test)

    pred = np.expm1(pred_log)

    pred = np.maximum(pred, 0)

    raw_mae = mean_absolute_error(
        y_test_actual,
        pred
    )

    raw_rmse = np.sqrt(
        mean_squared_error(
            y_test_actual,
            pred
        )
    )

    raw_r2 = r2_score(
        y_test_actual,
        pred
    )

    log_mae = mean_absolute_error(
        y_test_log,
        pred_log
    )

    log_rmse = np.sqrt(
        mean_squared_error(
            y_test_log,
            pred_log
        )
    )

    log_r2 = r2_score(
        y_test_log,
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

    results.append({
        "model": name,
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "raw_r2": raw_r2,
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "log_r2": log_r2,
    })


results_df = pd.DataFrame(results)

print()
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    results_df
    .sort_values("log_r2", ascending=False)
    .to_string(index=False)
)