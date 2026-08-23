import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
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
preprocessor = joblib.load("data/processed/preprocessor.joblib")

print("Transforming training data...")
X_train_processed = preprocessor.transform(X_train)
print("Training processed shape:", X_train_processed.shape)

print("\nTransforming test data...")
X_test_processed = preprocessor.transform(X_test)
print("Test processed shape:", X_test_processed.shape)

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

print("\nGenerating predictions...")

y_pred = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

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

cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)

joblib.dump(
    model,
    "data/processed/random_forest_model.joblib"
)

print("\nSaved Random Forest model:")
print("data/processed/random_forest_model.joblib")