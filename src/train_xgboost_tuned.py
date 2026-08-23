import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading data...")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\nLoading preprocessor...")

preprocessor = joblib.load(
    "data/processed/preprocessor.joblib"
)

print("Transforming training data...")

X_train_processed = preprocessor.transform(
    X_train
)

print(
    "Training processed shape:",
    X_train_processed.shape
)

print("\nTransforming test data...")

X_test_processed = preprocessor.transform(
    X_test
)

print(
    "Test processed shape:",
    X_test_processed.shape
)

# ---------------------------------------------------------
# Encode target
# ---------------------------------------------------------

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train
)

y_test_encoded = label_encoder.transform(
    y_test
)

print("\nTarget classes:")

for i, label in enumerate(label_encoder.classes_):
    print(i, "->", label)

# ---------------------------------------------------------
# Class weights
# ---------------------------------------------------------

class_counts = (
    pd.Series(y_train_encoded)
    .value_counts()
    .sort_index()
)

total_samples = len(y_train_encoded)
num_classes = len(class_counts)

class_weights = {
    cls: total_samples / (num_classes * count)
    for cls, count in class_counts.items()
}

sample_weights = pd.Series(
    y_train_encoded
).map(
    class_weights
).values

print("\nClass weights:")

for cls, weight in class_weights.items():
    print(
        label_encoder.classes_[cls],
        ":",
        round(weight, 4)
    )

# ---------------------------------------------------------
# Tuned XGBoost
# ---------------------------------------------------------

print("\nTraining tuned XGBoost...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,

    # More trees + smaller learning rate
    n_estimators=500,
    learning_rate=0.05,

    # Tree complexity
    max_depth=7,
    min_child_weight=3,

    # Randomization
    subsample=0.85,
    colsample_bytree=0.85,

    # Regularization
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,

    eval_metric="mlogloss",

    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model.fit(
    X_train_processed,
    y_train_encoded,
    sample_weight=sample_weights,
    eval_set=[
        (X_test_processed, y_test_encoded)
    ],
    verbose=False
)

print("\nGenerating predictions...")

y_pred_encoded = model.predict(
    X_test_processed
)

y_pred = label_encoder.inverse_transform(
    y_pred_encoded.astype(int)
)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("TUNED XGBOOST RESULTS")
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

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

joblib.dump(
    model,
    "data/processed/xgboost_tuned_model.joblib"
)

joblib.dump(
    label_encoder,
    "data/processed/xgboost_tuned_label_encoder.joblib"
)

print("\nSaved:")

print(
    "data/processed/xgboost_tuned_model.joblib"
)

print(
    "data/processed/xgboost_tuned_label_encoder.joblib"
)