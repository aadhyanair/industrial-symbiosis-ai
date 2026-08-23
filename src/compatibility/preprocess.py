"""
Preprocesses the compatibility dataset.
Fixes: Removes target-leaking features and ensures real feature names are saved.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib

INPUT_FILE = "data/processed/material_industry_compatibility.csv"
OUTPUT_DIR = "data/processed/"

def preprocess_data():
    print("Loading compatibility dataset...")
    df = pd.read_csv(INPUT_FILE)

    # Filter out empty classes
    df = df.dropna(subset=["compatibility_class"])

    # Define inputs and targets
    # EXCLUDED: recycling_rate, energy_recovery_rate, historical_utilization_score (Target Leakage)
    X = df.drop(columns=[
        "compatibility_score", 
        "compatibility_confidence", 
        "compatibility_class",
        "recycling_rate", 
        "energy_recovery_rate", 
        "historical_utilization_score"
    ])
    y = df["compatibility_class"]

    categorical_features = ["chemical_name", "primary_naics"]
    numerical_features = [
        "records", "total_release", "total_recycled", 
        "total_treated", "total_energy_recovery", 
        "total_managed", "treatment_rate", "management_rate"
    ]

    # Fill NaNs safely
    for col in numerical_features:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].fillna("UNKNOWN")

    X = X[categorical_features + numerical_features]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # EXPLICITLY GET FEATURE NAMES TO FIX INTEGER COLUMN ISSUE
    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the clean splits
    X_train_df.to_csv(os.path.join(OUTPUT_DIR, "X_train_compat.csv"), index=False)
    X_test_df.to_csv(os.path.join(OUTPUT_DIR, "X_test_compat.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train_compat.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test_compat.csv"), index=False)

    # Save the preprocessor
    os.makedirs("models/compatibility", exist_ok=True)
    joblib.dump(preprocessor, "models/compatibility/preprocessor.joblib")

    print(f"Preprocessing complete. Feature count: {len(feature_names)}")

if __name__ == "__main__":
    preprocess_data()