import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading training data...")

X = pd.read_csv("data/processed/X_train.csv")
y = pd.read_csv("data/processed/y_train.csv").squeeze()

print("Original training shape:", X.shape)

print("\nOriginal target distribution:")
print(y.value_counts())

# 80/20 split of existing training data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nNew training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

print("\nTraining distribution:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nValidation distribution:")
print(y_val.value_counts(normalize=True).mul(100).round(2))

# Save
X_train.to_csv(
    "data/processed/X_train_model.csv",
    index=False
)

X_val.to_csv(
    "data/processed/X_val.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train_model.csv",
    index=False
)

y_val.to_csv(
    "data/processed/y_val.csv",
    index=False
)

print("\nSaved:")
print("data/processed/X_train_model.csv")
print("data/processed/X_val.csv")
print("data/processed/y_train_model.csv")
print("data/processed/y_val.csv")