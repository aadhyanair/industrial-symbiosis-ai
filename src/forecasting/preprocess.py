"""
Preprocessing pipeline for waste-generation forecasting.

Two-stage forecasting:
1. Predict whether a material stream has positive waste release.
2. Predict log1p(quantity) for positive-release records.

Split is performed at facility level to prevent facility leakage.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = "data/processed/tri_waste_clean.csv"

X_TRAIN_OUT = "data/processed/forecasting_X_train.csv"
X_TEST_OUT = "data/processed/forecasting_X_test.csv"

Y_TRAIN_OUT = "data/processed/forecasting_y_train.csv"
Y_TEST_OUT = "data/processed/forecasting_y_test.csv"

PREPROCESSOR_OUT = "models/forecasting/preprocessor.joblib"

TARGET = "total_release_current"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT, low_memory=False)

print(f"Dataset shape: {df.shape}")

# ============================================================
# NAICS HIERARCHICAL FEATURES
# ============================================================

df["30_primary_naics_code"] = pd.to_numeric(
    df["30_primary_naics_code"],
    errors="coerce"
)

df["naics_2"] = (
    df["30_primary_naics_code"]
    .astype("Int64")
    .astype(str)
    .str[:2]
)

df["naics_3"] = (
    df["30_primary_naics_code"]
    .astype("Int64")
    .astype(str)
    .str[:3]
)

df["naics_4"] = (
    df["30_primary_naics_code"]
    .astype("Int64")
    .astype(str)
    .str[:4]
)
# ============================================================
# FEATURES
# ============================================================
#
# IMPORTANT:
# We deliberately exclude:
#
# - total_release_current       -> target
# - total_managed               -> derived from waste quantities
# - total_recycled              -> leakage
# - total_treated               -> leakage
# - total_energy_recovery       -> leakage
# - management_rate             -> leakage
# - release log                 -> leakage
# - recycled log                -> leakage
# - treated log                 -> leakage
# - managed log                 -> leakage
#
# Facility ID is used only for grouping/splitting.
# Facility name is excluded because it is high-cardinality.
#
# Source-reduction fields are excluded because >95% are missing.
# ============================================================

categorical_features = [
    "9_facility_state",
    "30_primary_naics_code",
    "naics_2",
    "naics_3",
    "naics_4",
    "68_cas_number",
    "70_chemical_name",
    "144_prod_ratio_or_activity",
]
numeric_features = [
    "36_latitude",
    "37_longitude",
    "143_prod_ratio_activity_index",
]

group_column = "4_trifd"


required_columns = (
    categorical_features
    + numeric_features
    + [group_column, TARGET]
)


missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df = df[required_columns].copy()


# ============================================================
# CLEAN TARGET
# ============================================================

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[TARGET, group_column]
)

# Negative releases should not exist.
# Protect against malformed values.
df[TARGET] = df[TARGET].clip(lower=0)


print(f"Rows used: {len(df):,}")


# ============================================================
# CREATE TWO FORECASTING TARGETS
# ============================================================

# Stage 1:
# 1 = positive release
# 0 = zero release

df["release_binary"] = (
    df[TARGET] > 0
).astype(int)


# Stage 2:
# log quantity for positive releases
#
# Zero-release rows receive 0 here.
# The training script will train the quantity model
# only on positive-release rows.

df["log_release"] = np.log1p(
    df[TARGET]
)


# ============================================================
# FACILITY-LEVEL TRAIN / TEST SPLIT
# ============================================================

print("\nPerforming facility-level train/test split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df[group_column]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()


# ============================================================
# VERIFY FACILITY LEAKAGE
# ============================================================

train_facilities = set(
    train_df[group_column]
)

test_facilities = set(
    test_df[group_column]
)

overlap = train_facilities.intersection(
    test_facilities
)

print(
    f"Training facilities: {len(train_facilities):,}"
)

print(
    f"Testing facilities:  {len(test_facilities):,}"
)

print(
    f"Facility overlap:    {len(overlap)}"
)

if overlap:
    raise RuntimeError(
        "Facility leakage detected!"
    )


# ============================================================
# PREPARE X / Y
# ============================================================

X_train = train_df[
    categorical_features + numeric_features
].copy()

X_test = test_df[
    categorical_features + numeric_features
].copy()


y_train = train_df[
    ["release_binary", "log_release", TARGET]
].copy()

y_test = test_df[
    ["release_binary", "log_release", TARGET]
].copy()


# ============================================================
# PREPROCESSOR
# ============================================================

print("\nBuilding preprocessing pipeline...")


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ============================================================
# FIT ONLY ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# FEATURE NAMES
# ============================================================

try:
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )
except Exception:
    feature_names = [
        f"feature_{i}"
        for i in range(
            X_train_processed.shape[1]
        )
    ]


X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

os.makedirs(
    "models/forecasting",
    exist_ok=True
)


X_train_processed.to_csv(
    X_TRAIN_OUT,
    index=False
)

X_test_processed.to_csv(
    X_TEST_OUT,
    index=False
)

y_train.to_csv(
    Y_TRAIN_OUT,
    index=False
)

y_test.to_csv(
    Y_TEST_OUT,
    index=False
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_OUT
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("WASTE FORECASTING PREPROCESSING")
print("=" * 60)

print(
    f"Training samples: {len(X_train_processed):,}"
)

print(
    f"Testing samples:  {len(X_test_processed):,}"
)

print(
    f"Processed features: {X_train_processed.shape[1]:,}"
)

print()
print("Training release distribution:")

print(
    y_train["release_binary"]
    .value_counts()
    .sort_index()
    .rename(
        {
            0: "ZERO_RELEASE",
            1: "POSITIVE_RELEASE",
        }
    )
    .to_string()
)

print()
print("Testing release distribution:")

print(
    y_test["release_binary"]
    .value_counts()
    .sort_index()
    .rename(
        {
            0: "ZERO_RELEASE",
            1: "POSITIVE_RELEASE",
        }
    )
    .to_string()
)

print()
print("Saved:")
print(X_TRAIN_OUT)
print(X_TEST_OUT)
print(Y_TRAIN_OUT)
print(Y_TEST_OUT)
print(PREPROCESSOR_OUT)