import pandas as pd
import numpy as np

real = pd.read_csv(
    "data/processed/material_industry_compatibility.csv"
)

synthetic = pd.read_csv(
    "data/processed/synthetic_waste_profiles.csv"
)

numeric = [
    "records",
    "total_release",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed",
    "recycling_rate",
    "treatment_rate",
    "energy_recovery_rate",
    "management_rate"
]

quantiles = [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]

print("\n========================================")
print("CTGAN V1 DISTRIBUTION EVALUATION")
print("========================================")

for col in numeric:

    real_q = real[col].quantile(quantiles)
    synthetic_q = synthetic[col].quantile(quantiles)

    print("\n" + col)
    print("-" * 70)
    print(
        f"{'Quantile':<12}"
        f"{'Real':>20}"
        f"{'Synthetic':>20}"
        f"{'Difference %':>17}"
    )

    for q in quantiles:

        r = real_q.loc[q]
        s = synthetic_q.loc[q]

        if abs(r) > 1e-12:
            diff = abs(r - s) / abs(r) * 100
        else:
            diff = 0 if abs(s) < 1e-12 else float("inf")

        print(
            f"{q:<12.2f}"
            f"{r:>20.4f}"
            f"{s:>20.4f}"
            f"{diff:>16.2f}%"
        )

print("\n========================================")
print("DISTRIBUTION EVALUATION COMPLETED")
print("========================================")