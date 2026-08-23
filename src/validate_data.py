import pandas as pd
import numpy as np

INPUT = "data/processed/tri_waste_clean.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT, low_memory=False)

print("\n" + "=" * 60)
print("1. BASIC INFORMATION")
print("=" * 60)

print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

print("\nMissing values:")
missing = df.isna().sum()
print(missing[missing > 0].sort_values(ascending=False).to_string())


print("\n" + "=" * 60)
print("2. NUMERIC FEATURE VALIDATION")
print("=" * 60)

numeric_cols = [
    "total_release_current",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed",
    "management_rate",
    "total_release_current_log",
    "total_recycled_log",
    "total_treated_log",
    "total_energy_recovery_log",
    "total_managed_log",
    "prod_ratio_activity_index",
]

for col in numeric_cols:
    if col not in df.columns:
        print(f"{col}: NOT FOUND")
        continue

    s = pd.to_numeric(df[col], errors="coerce")

    print(
        f"{col}: "
        f"min={s.min():.4f}, "
        f"max={s.max():.4f}, "
        f"negative={(s < 0).sum()}, "
        f"NaN={s.isna().sum()}, "
        f"inf={np.isinf(s).sum()}"
    )


print("\n" + "=" * 60)
print("3. MANAGEMENT RATE VALIDATION")
print("=" * 60)

rate = df["management_rate"]

print("Below 0:", (rate < 0).sum())
print("Above 1:", (rate > 1).sum())
print("Exactly 0:", (rate == 0).sum())
print("Exactly 1:", (rate == 1).sum())


print("\n" + "=" * 60)
print("4. UNIT VALIDATION")
print("=" * 60)

print(df["unit_of_measure"].value_counts(dropna=False).to_string())


print("\n" + "=" * 60)
print("5. RELEASE CONSISTENCY")
print("=" * 60)

release_components = [
    "103_on_site_limited_releases_current_year",
    "107_other_on_site_releases_current_year",
    "111_off_site_limited_releases_current_year",
    "115_other_off_site_releases_current_year",
]

calculated_release = (
    df[release_components]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .sum(axis=1)
)

difference = (
    df["total_release_current"] - calculated_release
).abs()

print("Maximum difference:", difference.max())
print("Rows with difference > 0.01:", (difference > 0.01).sum())


print("\n" + "=" * 60)
print("6. MANAGEMENT CONSISTENCY")
print("=" * 60)

management_components = [
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
]

calculated_managed = df[management_components].sum(axis=1)

difference = (
    df["total_managed"] - calculated_managed
).abs()

print("Maximum difference:", difference.max())
print("Rows with difference > 0.01:", (difference > 0.01).sum())


print("\n" + "=" * 60)
print("7. CATEGORICAL FEATURES")
print("=" * 60)

categorical_cols = [
    "chemical_name",
    "cas_number",
    "state",
    "facility_name",
    "primary_naics",
    "prod_ratio_activity",
]

for col in categorical_cols:
    if col in df.columns:
        print(
            f"{col}: "
            f"{df[col].nunique(dropna=True)} unique, "
            f"{df[col].isna().sum()} missing"
        )


print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
