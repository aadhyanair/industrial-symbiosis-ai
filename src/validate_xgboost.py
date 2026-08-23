import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading data...")

X_train = pd.read_csv(
    "data/processed/X_train_model.csv",
    low_memory=False
)

X_val = pd.read_csv(
    "data/processed/X_val.csv",
    low_memory=False
)

y_train = pd.read_csv(
    "data/processed/y_train_model.csv"
).squeeze()

y_val = pd.read_csv(
    "data/processed/y_val.csv"
).squeeze()

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)


# ============================================================
# 2. LOAD PREPROCESSOR
# ============================================================

print("\nLoading preprocessor...")

preprocessor = joblib.load(
    "data/processed/preprocessor.joblib"
)

print("Transforming training data...")

X_train_processed = preprocessor.transform(X_train)

print(
    "Training processed shape:",
    X_train_processed.shape
)

print("Transforming validation data...")

X_val_processed = preprocessor.transform(X_val)

print(
    "Validation processed shape:",
    X_val_processed.shape
)


# ============================================================
# 3. ENCODE TARGET
# ============================================================

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

y_train_encoded = y_train.map(label_to_int)
y_val_encoded = y_val.map(label_to_int)

print("\nTarget classes:")

for i, label in enumerate(classes):
    print(i, "->", label)


# ============================================================
# 4. CLASS WEIGHTS
# ============================================================

class_counts = y_train_encoded.value_counts().sort_index()

total = len(y_train_encoded)
n_classes = len(classes)

class_weights = {
    i: total / (n_classes * class_counts[i])
    for i in range(n_classes)
}

print("\nClass weights:")

for i, label in enumerate(classes):
    print(
        f"{label} : {class_weights[i]:.4f}"
    )


sample_weights = y_train_encoded.map(
    class_weights
).values


# ============================================================
# 5. TRAIN TUNED XGBOOST
# ============================================================

print("\nTraining tuned XGBoost on model-training split...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,

    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=3,
    gamma=0.1,

    reg_alpha=0.1,
    reg_lambda=1.0,

    eval_metric="mlogloss",

    tree_method="hist",

    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_processed,
    y_train_encoded,
    sample_weight=sample_weights
)


# ============================================================
# 6. VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

y_pred = model.predict(X_val_processed)


# ============================================================
# 7. RESULTS
# ============================================================

accuracy = accuracy_score(
    y_val_encoded,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_val_encoded,
    y_pred
)

print("\n" + "=" * 60)
print("VALIDATION XGBOOST RESULTS")
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
        y_val_encoded,
        y_pred,
        target_names=classes,
        digits=4
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_val_encoded,
    y_pred
)

print(
    pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )
)


# ============================================================
# 8. SAVE VALIDATION MODEL
# ============================================================

joblib.dump(
    model,
    "data/processed/xgboost_validation_model.joblib"
)

joblib.dump(
    classes,
    "data/processed/xgboost_validation_classes.joblib"
)

print(
    "\nSaved:"
)

print(
    "data/processed/xgboost_validation_model.joblib"
)

print(
    "data/processed/xgboost_validation_classes.joblib"
)