import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_X = "data/processed/X_train.csv"
PREPROCESSOR = "data/processed/preprocessor.joblib"

print("Loading training data...")

X_train = pd.read_csv(
    TRAIN_X,
    low_memory=False
)

print("Training shape:", X_train.shape)


# ============================================================
# FEATURE GROUPS
# ============================================================

numeric_features = [
    "total_release_current",
    "103_on_site_limited_releases_current_year",
    "107_other_on_site_releases_current_year",
    "111_off_site_limited_releases_current_year",
    "115_other_off_site_releases_current_year",
    "143_prod_ratio_activity_index",
    "36_latitude",
    "37_longitude",
]

categorical_features = [
    "9_facility_state",
    "30_primary_naics_code",
    "70_chemical_name",
    "144_prod_ratio_or_activity",
    "145_first_source_reduction_activity_code",
    "155_second_source_reduction_activity_code",
    "165_third_source_reduction_activity_code",
    "175_fourth_source_reduction_activity_code",
]


# ============================================================
# NUMERICAL PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="UNKNOWN"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# ============================================================
# COMBINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# FIT ONLY ON TRAINING DATA
# ============================================================

print("\nFitting preprocessor on training data...")

X_train_processed = preprocessor.fit_transform(X_train)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed matrix type:",
    type(X_train_processed)
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    preprocessor,
    PREPROCESSOR
)

print("\nSaved preprocessor:")
print(PREPROCESSOR)