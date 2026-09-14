import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

REAL = "data/processed/material_industry_compatibility.csv"
SYN = "data/processed/synthetic_waste_profiles_v5.csv"

qty_cols = [
    "records",
    "total_release",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed"
]

rate_cols = [
    "recycling_rate",
    "treatment_rate",
    "energy_recovery_rate",
    "management_rate"
]

cat_cols = [
    "chemical_name",
    "primary_naics"
]

real = pd.read_csv(REAL)
syn = pd.read_csv(SYN)

print("=" * 55)
print("V5 CTGAN DISTRIBUTION EVALUATION")
print("=" * 55)

# Zero percentage
print("\n--- Zero Percentage ---")
print(f"{'Feature':30} {'Real':>10} {'V5':>10} {'Diff':>10}")

for col in qty_cols:
    r = (real[col] == 0).mean() * 100
    s = (syn[col] == 0).mean() * 100
    print(f"{col:30} {r:10.2f} {s:10.2f} {abs(r-s):10.2f}")

# Quantiles
print("\n--- Quantile Comparison ---")

qs = [0.50, 0.75, 0.90, 0.95, 0.99]

for col in qty_cols:
    print(f"\n{col}")
    print(f"{'Quantile':10} {'Real':>18} {'V5':>18} {'Diff %':>12}")

    for q in qs:
        r = real[col][real[col] > 0].quantile(q)
        s = syn[col][syn[col] > 0].quantile(q)

        diff = abs(r - s) / max(abs(r), 1e-12) * 100

        print(f"{q:<10.2f} {r:18.3f} {s:18.3f} {diff:12.2f}")

# KS and Wasserstein
print("\n--- Distribution Metrics ---")
print(f"{'Feature':30} {'KS':>10} {'Wasserstein':>18}")

for col in qty_cols + rate_cols:
    r = real[col].values
    s = syn[col].values

    ks = ks_2samp(r, s).statistic
    wd = wasserstein_distance(r, s)

    print(f"{col:30} {ks:10.4f} {wd:18.4f}")

# Categorical similarity
print("\n--- Categorical Similarity ---")

for col in cat_cols:
    r = real[col].value_counts(normalize=True)
    s = syn[col].value_counts(normalize=True)

    index = r.index.union(s.index)

    r = r.reindex(index, fill_value=0)
    s = s.reindex(index, fill_value=0)

    similarity = 1 - 0.5 * np.abs(r - s).sum()

    print(f"{col}: {similarity:.4f}")

print("\n" + "=" * 55)
print("EVALUATION COMPLETED")
print("=" * 55)