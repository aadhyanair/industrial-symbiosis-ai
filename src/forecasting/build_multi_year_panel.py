"""
Build a multi-year panel dataset from raw TRI files (2021, 2022)
by summing up specific on-site and off-site release sub-categories.
"""

import os
import pandas as pd

print("Processing multi-year TRI datasets and summing release components...")

def load_tri_file(filepath, year):
    # Read header to dynamically match columns
    header_df = pd.read_csv(filepath, sep="\t", nrows=0, encoding="latin1")
    raw_cols = header_df.columns.tolist()

    # Identifiers
    id_targets = [
        "4. TRIFD", "9. FACILITY STATE", "30. PRIMARY NAICS CODE",
        "68. CAS NUMBER", "70. CHEMICAL NAME", "36. LATITUDE", "37. LONGITUDE",
        "143. PROD RATIO/ACTIVITY INDEX", "144. PROD RATIO OR ACTIVITY"
    ]
    
    # Release targets (we want both PRIOR/PREVIOUS and CURRENT)
    release_targets = [
        "ON-SITE LIMITED RELEASES",
        "OTHER ON-SITE RELEASES",
        "OFF-SITE LIMITED RELEASES",
        "OTHER OFF-SITE RELEASES"
    ]

    selected_cols = []
    for c in raw_cols:
        c_upper = c.upper()
        if any(t in c_upper for t in id_targets) or any(t in c_upper for t in release_targets):
            selected_cols.append(c)

    df = pd.read_csv(
        filepath,
        sep="\t",
        low_memory=False,
        encoding="latin1",
        usecols=selected_cols,
    )
    df.columns = [c.strip().upper() for c in df.columns]

    # Standardize identifiers
    rename_dict = {}
    for c in df.columns:
        if "TRIFD" in c: rename_dict[c] = "trifd"
        elif "FACILITY STATE" in c: rename_dict[c] = "facility_state"
        elif "PRIMARY NAICS CODE" in c: rename_dict[c] = "primary_naics_code"
        elif "CAS NUMBER" in c: rename_dict[c] = "cas_number"
        elif "CHEMICAL NAME" in c: rename_dict[c] = "chemical_name"
        elif "LATITUDE" in c: rename_dict[c] = "latitude"
        elif "LONGITUDE" in c: rename_dict[c] = "longitude"
        elif "PROD RATIO/ACTIVITY INDEX" in c: rename_dict[c] = "prod_ratio_activity_index"
        elif "PROD RATIO OR ACTIVITY" in c: rename_dict[c] = "prod_ratio_or_activity"

    df = df.rename(columns=rename_dict)
    df["year"] = year

    # Identify all current and prior release columns
    current_cols = [c for c in df.columns if "CURRENT" in c and "RELEASE" in c]
    prior_cols = [c for c in df.columns if ("PRIOR" in c or "PREVIOUS" in c) and "RELEASE" in c]

    # Convert all release columns to numeric safely
    for c in current_cols + prior_cols:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Sum them up into our target features
    df["quantity_current_year"] = df[current_cols].sum(axis=1)
    df["quantity_prior_year"] = df[prior_cols].sum(axis=1)

    # Keep only the columns we need
    final_cols = [
        "trifd", "facility_state", "primary_naics_code", "cas_number", 
        "chemical_name", "latitude", "longitude", "prod_ratio_activity_index", 
        "prod_ratio_or_activity", "quantity_prior_year", "quantity_current_year", "year"
    ]
    
    return df[final_cols]

# Load and combine
df_2021 = load_tri_file("data/raw/US_2A_2021.txt", 2021)
df_2022 = load_tri_file("data/raw/US_2A_2022.txt", 2022)

panel_df = pd.concat([df_2021, df_2022], ignore_index=True)

output_path = "data/processed/tri_multi_year_panel.csv"
os.makedirs("data/processed", exist_ok=True)
panel_df.to_csv(output_path, index=False)

print(f"\nPanel successfully created with {len(panel_df):,} rows!")
print(f"Non-zero current release records: {(panel_df['quantity_current_year'] > 0).sum():,}")
print(f"Saved to: {output_path}")