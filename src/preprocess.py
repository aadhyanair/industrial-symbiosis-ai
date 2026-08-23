import pandas as pd
import numpy as np

INPUT = "data/raw/tri_waste_management_2023_corrected.csv"
OUTPUT = "data/processed/tri_waste_clean.csv"

print("Loading dataset...")

df = pd.read_csv(
    INPUT,
    low_memory=False
)

print("Original shape:", df.shape)

# --------------------------------------------------
# 1. Remove exact duplicate rows
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Exact duplicates removed:", before - after)
print("New shape:", df.shape)


# --------------------------------------------------
# 2. Standardize column names
# --------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)


# --------------------------------------------------
# 3. Normalize units
# --------------------------------------------------

df["unit_of_measure"] = (
    df["74_unit_of_measure"]
    .astype(str)
    .str.strip()
    .str.upper()
)

print("\nUnits before conversion:")
print(df["unit_of_measure"].value_counts(dropna=False))


# Quantity columns that need unit normalization
quantity_cols = [
    "total_release_current",

    "103_on_site_limited_releases_current_year",
    "107_other_on_site_releases_current_year",
    "111_off_site_limited_releases_current_year",
    "115_other_off_site_releases_current_year",

    "119_energy_recovery_on_site_current_year",
    "123_energy_recovery_off_site_current_year",

    "127_quantity_recycled_on_site_current_year",
    "131_quantity_recycled_off_site_current_year",

    "135_quantity_treated_on_site_current_year",
    "139_quantity_treated_off_site_current_year",
]


# Convert numeric columns
for col in quantity_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Convert GRAMS → POUNDS
gram_mask = df["unit_of_measure"] == "GRAMS"

print("\nRows requiring grams → pounds conversion:", gram_mask.sum())

for col in quantity_cols:
    df.loc[gram_mask, col] = (
        df.loc[gram_mask, col] / 453.59237
    )


# Everything is now represented in pounds
df["unit_of_measure"] = "POUNDS"

# --------------------------------------------------
# 4. Create waste-management aggregate features
# --------------------------------------------------

df["total_recycled"] = (
    df["127_quantity_recycled_on_site_current_year"].fillna(0)
    + df["131_quantity_recycled_off_site_current_year"].fillna(0)
)

df["total_treated"] = (
    df["135_quantity_treated_on_site_current_year"].fillna(0)
    + df["139_quantity_treated_off_site_current_year"].fillna(0)
)

df["total_energy_recovery"] = (
    df["119_energy_recovery_on_site_current_year"].fillna(0)
    + df["123_energy_recovery_off_site_current_year"].fillna(0)
)

df["total_managed"] = (
    df["total_recycled"]
    + df["total_treated"]
    + df["total_energy_recovery"]
)


# --------------------------------------------------
# 5. Create management rate
# --------------------------------------------------

df["management_rate"] = (
    df["total_managed"]
    / (
        df["total_managed"]
        + df["total_release_current"]
        + 1e-9
    )
)

print("\nNew waste-management features:")

feature_cols = [
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed",
    "management_rate"
]

print(df[feature_cols].describe().T.to_string())
# --------------------------------------------------
# 6. Log-transform highly skewed quantity features
# --------------------------------------------------

log_cols = [
    "total_release_current",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed"
]

for col in log_cols:
    df[col + "_log"] = np.log1p(df[col])

print("\nLog-transformed feature statistics:")

log_feature_cols = [col + "_log" for col in log_cols]

print(
    df[log_feature_cols]
    .describe()
    .T
    .to_string()
)
# --------------------------------------------------
# 7. Clean categorical and identifier fields
# --------------------------------------------------

df["chemical_name"] = (
    df["70_chemical_name"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
)

df["state"] = (
    df["9_facility_state"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
    .str.upper()
)

df["facility_name"] = (
    df["5_facility_name"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
)

df["cas_number"] = (
    df["68_cas_number"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
)

df["primary_naics"] = (
    df["30_primary_naics_code"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
)

df["prod_ratio_activity"] = (
    df["144_prod_ratio_or_activity"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
    .str.upper()
)

print("\nCategorical feature summary:")

for col in [
    "chemical_name",
    "state",
    "facility_name",
    "cas_number",
    "primary_naics",
    "prod_ratio_activity"
]:
    print(
        f"{col}:",
        df[col].nunique(),
        "unique values"
    )
# --------------------------------------------------
# 8. Save cleaned dataset
# --------------------------------------------------

df.to_csv(OUTPUT, index=False)

print("\nUnits after conversion:")
print(df["unit_of_measure"].value_counts(dropna=False))

print("\nSaved:", OUTPUT)
print("Final shape:", df.shape)