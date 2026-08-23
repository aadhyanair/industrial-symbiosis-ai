import pandas as pd
import numpy as np

INPUT = "data/processed/tri_waste_clean.csv"
OUTPUT = "data/processed/material_industry_compatibility.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT, low_memory=False)

print(f"Dataset shape: {df.shape}")

# ---------------------------------------------------------
# Required columns
# ---------------------------------------------------------

cols = [
    "chemical_name",
    "primary_naics",
    "total_release_current",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
]

df = df[cols].copy()

# ---------------------------------------------------------
# Remove rows without material / industry information
# ---------------------------------------------------------

df = df.dropna(
    subset=["chemical_name", "primary_naics"]
)

print(
    f"Rows after removing missing material/industry: {len(df):,}"
)

# ---------------------------------------------------------
# Aggregate material-industry relationships
# ---------------------------------------------------------

grouped = (
    df.groupby(
        ["chemical_name", "primary_naics"],
        as_index=False
    )
    .agg(
        records=("chemical_name", "size"),
        total_release=("total_release_current", "sum"),
        total_recycled=("total_recycled", "sum"),
        total_treated=("total_treated", "sum"),
        total_energy_recovery=("total_energy_recovery", "sum"),
    )
)

# ---------------------------------------------------------
# Total managed
# ---------------------------------------------------------

grouped["total_managed"] = (
    grouped["total_recycled"]
    + grouped["total_treated"]
    + grouped["total_energy_recovery"]
)

# ---------------------------------------------------------
# Management rates
# ---------------------------------------------------------

denominator = grouped["total_release"].replace(0, np.nan)

grouped["recycling_rate"] = (
    grouped["total_recycled"] / denominator
)

grouped["treatment_rate"] = (
    grouped["total_treated"] / denominator
)

grouped["energy_recovery_rate"] = (
    grouped["total_energy_recovery"] / denominator
)

# ---------------------------------------------------------
# Management rate
#
# Bound to [0, 1].
# ---------------------------------------------------------

grouped["management_rate"] = (
    grouped["total_managed"]
    / (
        grouped["total_managed"]
        + grouped["total_release"]
    ).replace(0, np.nan)
)

# ---------------------------------------------------------
# Invalid / missing rates
# ---------------------------------------------------------

rate_cols = [
    "recycling_rate",
    "treatment_rate",
    "energy_recovery_rate",
    "management_rate",
]

for col in rate_cols:
    grouped[col] = (
        grouped[col]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(0, 1)
    )

# ---------------------------------------------------------
# Historical utilization score
#
# Recycling = strongest positive signal
# Energy recovery = secondary positive signal
# Treatment = not counted as reuse
# ---------------------------------------------------------

grouped["historical_utilization_score"] = (
    0.70 * grouped["recycling_rate"]
    + 0.30 * grouped["energy_recovery_rate"]
)

# ---------------------------------------------------------
# Historical evidence confidence
#
# A relationship observed once should not have the same
# confidence as a relationship observed many times.
#
# Bayesian-style smoothing:
#
# confidence = records / (records + K)
#
# K = 5 means approximately:
#   1 record  -> 0.167 confidence
#   5 records -> 0.500 confidence
#   20 records -> 0.800 confidence
#   100 records -> 0.952 confidence
# ---------------------------------------------------------

K = 5.0

grouped["compatibility_confidence"] = (
    grouped["records"]
    / (grouped["records"] + K)
)

# ---------------------------------------------------------
# Confidence-adjusted compatibility score
# ---------------------------------------------------------

grouped["compatibility_score"] = (
    grouped["historical_utilization_score"]
    * grouped["compatibility_confidence"]
)

# ---------------------------------------------------------
# Compatibility class
#
# Used later for ML training.
# ---------------------------------------------------------

grouped["compatibility_class"] = pd.cut(
    grouped["compatibility_score"],
    bins=[-np.inf, 0.25, 0.60, np.inf],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
    ],
)

# ---------------------------------------------------------
# Sort
# ---------------------------------------------------------

grouped = grouped.sort_values(
    "compatibility_score",
    ascending=False,
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

grouped.to_csv(
    OUTPUT,
    index=False,
)

# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------

print()
print("=" * 70)
print("MATERIAL-INDUSTRY COMPATIBILITY ANALYSIS")
print("=" * 70)

print(
    f"Unique chemicals: "
    f"{grouped['chemical_name'].nunique():,}"
)

print(
    f"Unique industries: "
    f"{grouped['primary_naics'].nunique():,}"
)

print(
    f"Material-industry pairs: "
    f"{len(grouped):,}"
)

print()
print("Rate sanity checks:")

for col in rate_cols:
    print(
        f"{col:28s}"
        f"min={grouped[col].min():.4f}  "
        f"max={grouped[col].max():.4f}  "
        f"mean={grouped[col].mean():.4f}"
    )

print()
print("Compatibility class distribution:")

print(
    grouped["compatibility_class"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("Top material-industry relationships:")

display_cols = [
    "chemical_name",
    "primary_naics",
    "records",
    "total_release",
    "total_managed",
    "recycling_rate",
    "treatment_rate",
    "energy_recovery_rate",
    "management_rate",
    "historical_utilization_score",
    "compatibility_confidence",
    "compatibility_score",
    "compatibility_class",
]

print(
    grouped[display_cols]
    .head(20)
    .to_string(index=False)
)

print()
print(f"Saved: {OUTPUT}")