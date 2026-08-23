import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/processed/tri_waste_features.csv"

TRAIN_X = "data/processed/X_train.csv"
TEST_X = "data/processed/X_test.csv"
TRAIN_Y = "data/processed/y_train.csv"
TEST_Y = "data/processed/y_test.csv"

print("Loading feature dataset...")

df = pd.read_csv(INPUT, low_memory=False)

print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# Separate features and target
# ------------------------------------------------------------

X = df.drop(columns=["management_target"])
y = df["management_target"]

print("\nFull target distribution:")
print(y.value_counts().to_string())

# ------------------------------------------------------------
# Stratified 80/20 split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# Print shapes
# ------------------------------------------------------------

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts().to_string())

print("\nTraining percentages:")
print(
    (y_train.value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)

print("\nTest target distribution:")
print(y_test.value_counts().to_string())

print("\nTest percentages:")
print(
    (y_test.value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

X_train.to_csv(TRAIN_X, index=False)
X_test.to_csv(TEST_X, index=False)

y_train.to_csv(TRAIN_Y, index=False)
y_test.to_csv(TEST_Y, index=False)

print("\nSaved:")
print(TRAIN_X)
print(TEST_X)
print(TRAIN_Y)
print(TEST_Y)