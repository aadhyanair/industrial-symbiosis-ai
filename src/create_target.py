import pandas as pd

INPUT = "data/processed/tri_waste_clean.csv"
OUTPUT = "data/processed/tri_waste_target.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT, low_memory=False)

# Waste-management quantities
management_cols = [
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
]

# Total amount managed through the three strategies
management_total = df[management_cols].sum(axis=1)

# Determine dominant management strategy
target = df[management_cols].idxmax(axis=1)

# Records with no management activity
target = target.where(
    management_total > 0,
    "no_management"
)

# Convert column names to clean target labels
target = target.replace({
    "total_recycled": "recycling",
    "total_treated": "treatment",
    "total_energy_recovery": "energy_recovery",
})

# Add target
df["management_target"] = target

print("\nTarget distribution:")
print(
    df["management_target"]
    .value_counts()
    .to_string()
)

print("\nTarget percentages:")
print(
    (df["management_target"].value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)

# Save
df.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("Shape:", df.shape)