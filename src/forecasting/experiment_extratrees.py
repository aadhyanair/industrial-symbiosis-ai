import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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

# ------------------------------------------------------------
# Positive-release samples only
# ------------------------------------------------------------

train_mask = y_train["release_binary"] == 1
test_mask = y_test["release_binary"] == 1

X_train_pos = X_train.loc[train_mask]
X_test_pos = X_test.loc[test_mask]

y_train_log = y_train.loc[
    train_mask, "log_release"
]

y_test_actual = y_test.loc[
    test_mask, "total_release_current"
]

# ------------------------------------------------------------
# Extra Trees
# ------------------------------------------------------------

print("Training Extra Trees Regressor...")

model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train_pos,
    y_train_log
)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

pred_log = model.predict(X_test_pos)

pred_quantity = np.expm1(pred_log)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

actual_log = np.log1p(
    y_test_actual
)

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
print("EXTRA TREES QUANTITY REGRESSION")
print("=" * 60)

print(f"Raw MAE:   {raw_mae:,.2f}")
print(f"Raw RMSE:  {raw_rmse:,.2f}")
print(f"Raw R2:    {raw_r2:.4f}")

print()

print(f"Log MAE:   {log_mae:.4f}")
print(f"Log RMSE:  {log_rmse:.4f}")
print(f"Log R2:    {log_r2:.4f}")