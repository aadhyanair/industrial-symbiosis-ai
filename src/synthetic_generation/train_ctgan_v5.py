import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from ctgan import CTGAN

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(BASE, "data", "processed", "material_industry_compatibility.csv")
MODEL = os.path.join(BASE, "models", "ctgan", "waste_profile_ctgan_v5.pkl")
OUTPUT = os.path.join(BASE, "data", "processed", "synthetic_waste_profiles_v5.csv")

cat_cols = ["chemical_name", "primary_naics"]

qty_cols = [
    "records",
    "total_release",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed"
]

zero_cols = [
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

df = pd.read_csv(INPUT)

data = df[cat_cols + qty_cols + rate_cols].copy()

# Store empirical mappings for positive values
mappings = {}

# Transform quantity columns
for col in qty_cols:
    if col in zero_cols:
        data[col + "_zero"] = (data[col] == 0).astype(int)

    values = data[col].values
    positive = values[values > 0]

    if len(positive) > 0:
        sorted_values = np.sort(positive)
        ranks = np.searchsorted(sorted_values, positive, side="right")
        probs = (ranks - 0.5) / len(sorted_values)
        probs = np.clip(probs, 0.0001, 0.9999)

        mapping_values = np.unique(sorted_values)
        mapping_probs = np.searchsorted(sorted_values, mapping_values, side="right")
        mapping_probs = (mapping_probs - 0.5) / len(sorted_values)
        mapping_probs = np.clip(mapping_probs, 0.0001, 0.9999)

        mappings[col] = (mapping_values, mapping_probs)

        transformed = np.zeros(len(values))
        positive_mask = values > 0
        transformed[positive_mask] = norm.ppf(
            np.interp(
                values[positive_mask],
                mapping_values,
                mapping_probs
            )
        )
        data[col] = transformed

# Keep rates valid
for col in rate_cols:
    data[col] = data[col].clip(0, 1)

model = CTGAN(
    epochs=300,
    batch_size=500,
    verbose=True
)

model.fit(
    data,
    discrete_columns=cat_cols + [c + "_zero" for c in zero_cols]
)

os.makedirs(os.path.dirname(MODEL), exist_ok=True)
model.save(MODEL)

synthetic = model.sample(len(df))

# Inverse-transform quantity columns
for col in qty_cols:
    values = synthetic[col].values

    if col in zero_cols:
        zero = synthetic[col + "_zero"].round().clip(0, 1).astype(int)
    else:
        zero = np.zeros(len(synthetic), dtype=int)

    mapping_values, mapping_probs = mappings[col]

    generated_probs = norm.cdf(values)
    generated_probs = np.clip(generated_probs, 0.0001, 0.9999)

    restored = np.interp(
        generated_probs,
        mapping_probs,
        mapping_values
    )

    if col in zero_cols:
        restored[zero == 1] = 0

    synthetic[col] = np.maximum(restored, 0)

    if col == "records":
        synthetic[col] = np.round(synthetic[col]).astype(int)

    if col in zero_cols:
        synthetic.drop(columns=[col + "_zero"], inplace=True)

# Keep rates valid
for col in rate_cols:
    synthetic[col] = synthetic[col].clip(0, 1)

# Restore original column order
synthetic = synthetic[cat_cols + qty_cols + rate_cols]

synthetic.to_csv(OUTPUT, index=False)

print("\nSynthetic data generated successfully.")
print("Shape:", synthetic.shape)
print("Model:", MODEL)
print("Output:", OUTPUT)