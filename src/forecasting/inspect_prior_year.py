Set-Content -Path "src/forecasting/inspect_prior_year.py" -Value '"""
Inspect prior and current release fields across multi-year TRI data.
"""

import pandas as pd

file_path = "data/raw/US_2A_2021.txt"

print("Loading sample from 2021 file...")
df = pd.read_csv(
    file_path, 
    sep="\t", 
    low_memory=False, 
    encoding="latin1", 
    usecols=[
        "4. TRIFD", 
        "68. CAS NUMBER", 
        "98. QUANTITY RELEASED PRIOR YEAR", 
        "99. QUANTITY RELEASED CURRENT YEAR"
    ]
)

print("Shape:", df.shape)
print("\nMissing values check:")
print(df.isna().sum())

print("\nSample rows:")
print(df.dropna().head(10).to_string())
'