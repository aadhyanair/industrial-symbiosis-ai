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


print("Loading data...")

X_train = pd.read_csv(X_TRAIN)
X_test = pd.read_csv(X_TEST)

y_train = pd.read_csv(Y_TRAIN)
y_test = pd.read_csv(Y_TEST)


# Positive-release records only
train_mask = y_train["release_binary"] == 1
test_mask = y_test["release_binary"] == 1

X_train_positive = X_train.loc[train_mask]
X_test_positive = X_test.loc[test_mask]

y_train_log = y_train.loc[
    train_mask, "log_release"
]

y_test_actual = y_test.loc[
    test_mask, "total_release_current"
]


print("Training samples:", len(X_train_positive))
print("Testing samples:", len(X_test_positive))


print("\nTraining HistGradientBoosting Regressor...")


model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)


model.fit(
    X_train_positive,
    y_train_log
)


# Predict log quantity
pred_log = model.predict(X_test_positive)

# Convert back to pounds
pred_quantity = np.expm1(pred_log)


# Actual log quantity
actual_log = np.log1p(y_test_actual)


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
print("=" * 60)
print("HISTGRADIENTBOOSTING QUANTITY REGRESSION")
print("=" * 60)

print(f"Raw MAE:   {raw_mae:,.2f}")
print(f"Raw RMSE:  {raw_rmse:,.2f}")
print(f"Raw R2:    {raw_r2:.4f}")

print()

print(f"Log MAE:   {log_mae:.4f}")
print(f"Log RMSE:  {log_rmse:.4f}")
print(f"Log R2:    {log_r2:.4f}")