import pandas as pd
import numpy as np

INPUT = "data/processed/tri_waste_target.csv"
OUTPUT = "data/processed/tri_waste_features.csv"

print("Loading target dataset...")
df = pd.read_csv(INPUT, low_memory=False)

print("Original shape:", df.shape)

# ============================================================
# 1. SELECT FEATURES
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

target = "management_target"

selected = numeric_features + categorical_features + [target]

missing_columns = [c for c in selected if c not in df.columns]

if missing_columns:
    print("\nERROR: Missing columns:")
    for c in missing_columns:
        print(" -", c)
    raise ValueError("Required columns are missing.")

features = df[selected].copy()

# ============================================================
# 2. CLEAN NUMERIC FEATURES
# ============================================================

for col in numeric_features:
    features[col] = pd.to_numeric(
        features[col],
        errors="coerce"
    )

    # Replace infinite values
    features[col] = features[col].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Median imputation
    features[col] = features[col].fillna(
        features[col].median()
    )

# ============================================================
# 3. CLEAN CATEGORICAL FEATURES
# ============================================================

for col in categorical_features:
    features[col] = (
        features[col]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

# ============================================================
# 4. REMOVE ROWS WITH MISSING TARGET
# ============================================================

before = len(features)

features = features.dropna(
    subset=[target]
).reset_index(drop=True)

removed = before - len(features)

print("\nRows removed because of missing target:", removed)

# ============================================================
# 5. TARGET VALIDATION
# ============================================================

print("\nTarget distribution:")
print(features[target].value_counts().to_string())

print("\nTarget percentages:")
print(
    (features[target].value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)

# ============================================================
# 6. SAVE
# ============================================================

features.to_csv(
    OUTPUT,
    index=False
)

print("\nSelected numeric features:")
for c in numeric_features:
    print(" -", c)

print("\nSelected categorical features:")
for c in categorical_features:
    print(" -", c)

print("\nFinal shape:", features.shape)

print("\nSaved:", OUTPUT)