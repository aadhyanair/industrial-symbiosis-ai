
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================================
# LOAD DATA
# ============================================================

TRAIN = "data/processed/forecasting_y_train.csv"
TEST = "data/processed/forecasting_predictions.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

# Only positive-release records
train_pos = train[train["release_binary"] == 1]
test_pos = test[test["actual_release_binary"] == 1]

y_train = train_pos["total_release_current"]
y_test = test_pos["actual_quantity"]

log_y_train = np.log1p(y_train)
log_y_test = np.log1p(y_test)

print("=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

# ============================================================
# BASELINES
# ============================================================

# 1. Median in raw space
raw_median = y_train.median()
raw_pred = np.full(len(y_test), raw_median)

# 2. Mean in raw space
raw_mean = y_train.mean()
mean_pred = np.full(len(y_test), raw_mean)

# 3. Median in log space
log_median = log_y_train.median()
log_pred = np.expm1(np.full(len(y_test), log_median))

# 4. Mean in log space
log_mean = log_y_train.mean()
log_mean_pred = np.expm1(np.full(len(y_test), log_mean))


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(name, prediction):

    raw_mae = mean_absolute_error(
        y_test,
        prediction
    )

    raw_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    raw_r2 = r2_score(
        y_test,
        prediction
    )

    log_mae = mean_absolute_error(
        log_y_test,
        np.log1p(prediction)
    )

    log_rmse = np.sqrt(
        mean_squared_error(
            log_y_test,
            np.log1p(prediction)
        )
    )

    log_r2 = r2_score(
        log_y_test,
        np.log1p(prediction)
    )

    print()
    print(name)
    print("-" * 40)
    print(f"Raw MAE:   {raw_mae:,.2f}")
    print(f"Raw RMSE:  {raw_rmse:,.2f}")
    print(f"Raw R2:    {raw_r2:.4f}")
    print(f"Log MAE:   {log_mae:.4f}")
    print(f"Log RMSE:  {log_rmse:.4f}")
    print(f"Log R2:    {log_r2:.4f}")


# ============================================================
# RESULTS
# ============================================================

evaluate(
    "RAW MEDIAN BASELINE",
    raw_pred
)

evaluate(
    "RAW MEAN BASELINE",
    mean_pred
)

evaluate(
    "LOG MEDIAN BASELINE",
    log_pred
)

evaluate(
    "LOG MEAN BASELINE",
    log_mean_pred
)

# Existing HistGradientBoosting predictions
hgb_prediction = test_pos["positive_quantity_prediction"].values

evaluate(
    "HISTGRADIENTBOOSTING",
    hgb_prediction
)

# ============================================================
# VALUES
# ============================================================

print()
print("=" * 60)
print("BASELINE VALUES")
print("=" * 60)

print(f"Raw median: {raw_median:,.4f}")
print(f"Raw mean:   {raw_mean:,.4f}")
print(f"Log median: {log_median:.4f}")
print(f"Log mean:   {log_mean:.4f}")

print()
print("Positive training samples:", len(train_pos))
print("Positive testing samples: ", len(test_pos))

