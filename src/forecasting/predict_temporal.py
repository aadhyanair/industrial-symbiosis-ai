"""
Generate final temporal forecasts (2022) using the trained 2021 model.
These outputs feed directly into the Compatibility and Matching modules.
"""

import os
import joblib
import numpy as np
import pandas as pd

INPUT = "data/processed/tri_multi_year_panel.csv"
OUTPUT = "data/processed/temporal_forecast_results.csv"

def generate_forecasts():
    print("Loading temporal forecasting models...")
    classifier = joblib.load("models/forecasting_temporal/classifier.joblib")
    regressor = joblib.load("models/forecasting_temporal/regressor.joblib")
    preprocessor = joblib.load("models/forecasting_temporal/preprocessor.joblib")

    print("Loading 2022 target data...")
    df = pd.read_csv(INPUT, low_memory=False)
    
    # We forecast for 2022 using 2021 data as the prior year lag
    test_df = df[df["year"] == 2022].copy()
    
    features = [
        "facility_state", "primary_naics_code", "cas_number", 
        "chemical_name", "prod_ratio_or_activity", "latitude", 
        "longitude", "prod_ratio_activity_index", "quantity_prior_year"
    ]
    
    X_test = test_df[features].copy()
    
    print("Processing features...")
    X_test_processed = preprocessor.transform(X_test)
    
    print("Generating predictions...")
    # Stage 1: Probability of release
    release_prob = classifier.predict_proba(X_test_processed)[:, 1]
    
    # Stage 2: Expected quantity if positive
    predicted_log_qty = regressor.predict(X_test_processed)
    predicted_positive_qty = np.expm1(predicted_log_qty)
    
    # Final Expected Forecast = Probability * Positive Quantity
    expected_forecast = release_prob * predicted_positive_qty
    
    # Save the results
    results_df = test_df[["trifd", "chemical_name", "primary_naics_code", "quantity_current_year"]].copy()
    results_df.rename(columns={"quantity_current_year": "actual_quantity"}, inplace=True)
    results_df["forecasted_quantity"] = expected_forecast
    results_df["release_probability"] = release_prob
    
    results_df.to_csv(OUTPUT, index=False)
    print(f"\nForecast successfully generated for {len(results_df):,} records!")
    print(f"Saved to: {OUTPUT}")

if __name__ == "__main__":
    generate_forecasts()