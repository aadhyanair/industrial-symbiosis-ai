import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score
)


# ============================================================
# PATHS
# ============================================================

TRAIN_X = "data/processed/X_train.csv"
TEST_X = "data/processed/X_test.csv"

TRAIN_Y = "data/processed/y_train.csv"
TEST_Y = "data/processed/y_test.csv"

PREPROCESSOR = "data/processed/preprocessor.joblib"

MODEL_PATH = "data/processed/baseline_model.joblib"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

X_train = pd.read_csv(
    TRAIN_X,
    low_memory=False
)

X_test = pd.read_csv(
    TEST_X,
    low_memory=False
)

y_train = pd.read_csv(TRAIN_Y)["management_target"]
y_test = pd.read_csv(TEST_Y)["management_target"]

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

print("\nLoading preprocessor...")

preprocessor = joblib.load(
    PREPROCESSOR
)


# ============================================================
# TRANSFORM DATA
# ============================================================

print("Transforming training data...")

X_train_processed = preprocessor.transform(
    X_train
)

print("Training processed shape:", X_train_processed.shape)

print("\nTransforming test data...")

X_test_processed = preprocessor.transform(
    X_test
)

print("Test processed shape:", X_test_processed.shape)


# ============================================================
# TRAIN BASELINE
# ============================================================

print("\nTraining Logistic Regression baseline...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="saga",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_processed,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test_processed
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("BASELINE RESULTS")
print("=" * 60)

print(
    f"Accuracy:          {accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


print("\nConfusion Matrix:")

labels = [
    "no_management",
    "treatment",
    "recycling",
    "energy_recovery"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\nSaved baseline model:")
print(MODEL_PATH)