"""
End-to-end Person A pipeline (demo-ready for viva/presentation).

    Facility + material inputs
            |
            v
    Waste Forecasting (Stage 1 + Stage 2 Temporal Models)
            |
            v
    Forecasted expected waste quantity (lbs)
            |
            v
    Compatibility Matching (historical compatibility scores)
            |
            v
    Ranked receiving industries
            |
            v
    Impact Estimation (CO2e avoided + economic impact)
            |
            v
    Final recommendation report

Run from the project root:

    python src/pipeline.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Make sibling packages (matching, impact) importable when running
# this file directly with `python src/pipeline.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matching.match_scorer import recommend_industries          # noqa: E402
from impact.impact_calculator import score_matches_with_impact  # noqa: E402


# ============================================================
# CONFIG — paths to the final temporal forecasting artifacts
# ============================================================

FORECAST_PREPROCESSOR = "models/forecasting_temporal/preprocessor.joblib"
FORECAST_CLASSIFIER = "models/forecasting_temporal/classifier.joblib"
FORECAST_REGRESSOR = "models/forecasting_temporal/regressor.joblib"

# ============================================================
# STAGE 1: WASTE FORECASTING
# ============================================================

def forecast_expected_quantity(
    facility_state,
    primary_naics_code,
    cas_number,
    chemical_name,
    prod_ratio_or_activity,
    latitude,
    longitude,
    prod_ratio_activity_index,
    quantity_prior_year=0.0, # Added for temporal lag
):
    """
    Predict the expected waste release quantity (lbs) for a given
    facility/material combination, using the two-stage forecasting model.
    """
    preprocessor = joblib.load(FORECAST_PREPROCESSOR)
    classifier = joblib.load(FORECAST_CLASSIFIER)
    regressor = joblib.load(FORECAST_REGRESSOR)

    row = pd.DataFrame([{
        "facility_state": facility_state,
        "primary_naics_code": primary_naics_code,
        "cas_number": cas_number,
        "chemical_name": chemical_name,
        "prod_ratio_or_activity": prod_ratio_or_activity,
        "latitude": latitude,
        "longitude": longitude,
        "prod_ratio_activity_index": prod_ratio_activity_index,
        "quantity_prior_year": quantity_prior_year,
    }])

    processed_array = preprocessor.transform(row)

    # CRITICAL FIX: Wrap back into a DataFrame with fitted feature names 
    # to eliminate UserWarnings during model prediction.
    try:
        feature_names = preprocessor.get_feature_names_out()
        processed_df = pd.DataFrame(processed_array, columns=feature_names)
    except Exception:
        processed_df = processed_array

    release_probability = classifier.predict_proba(processed_array)[0, 1]

    predicted_log_quantity = regressor.predict(processed_array)[0]
    predicted_quantity = max(np.expm1(predicted_log_quantity), 0.0)

    expected_quantity = release_probability * predicted_quantity

    return {
        "release_probability": float(release_probability),
        "predicted_quantity_if_release": float(predicted_quantity),
        "expected_quantity": float(expected_quantity),
    }

# ============================================================
# FULL PIPELINE: FORECAST -> MATCH -> IMPACT
# ============================================================

def run_pipeline(
    facility_state,
    primary_naics_code,
    cas_number,
    chemical_name,
    prod_ratio_or_activity,
    latitude,
    longitude,
    prod_ratio_activity_index,
    quantity_prior_year=0.0,
    top_k=5,
    min_records=3,
):

    print("=" * 70)
    print("STEP 1 — WASTE GENERATION FORECASTING")
    print("=" * 70)

    forecast = forecast_expected_quantity(
        facility_state=facility_state,
        primary_naics_code=primary_naics_code,
        cas_number=cas_number,
        chemical_name=chemical_name,
        prod_ratio_or_activity=prod_ratio_or_activity,
        latitude=latitude,
        longitude=longitude,
        prod_ratio_activity_index=prod_ratio_activity_index,
        quantity_prior_year=quantity_prior_year,
    )

    print(f"Chemical:                 {chemical_name}")
    print(f"Facility state:           {facility_state}")
    print(f"NAICS code:               {primary_naics_code}")
    print(f"Prior Year Waste:         {quantity_prior_year:,.2f} lbs")
    print(f"Release probability:      {forecast['release_probability']:.4f}")
    print(f"Predicted qty if release: {forecast['predicted_quantity_if_release']:,.2f} lbs")
    print(f"Expected quantity:        {forecast['expected_quantity']:,.2f} lbs")

    print()
    print("=" * 70)
    print("STEP 2 — COMPATIBILITY MATCHING")
    print("=" * 70)

    matches = recommend_industries(
        chemical_name=chemical_name,
        min_records=min_records,
        top_k=top_k,
    )

    if matches.empty:
        print(f"\nNo historical compatibility data found for '{chemical_name}'.")
        return {"forecast": forecast, "matches": matches}

    print(f"\nTop {len(matches)} candidate receiving industries:\n")
    print(matches[["primary_naics", "records", "compatibility_class", "compatibility_score", "match_score"]].to_string(index=False))

    print()
    print("=" * 70)
    print("STEP 3 — IMPACT ESTIMATION")
    print("=" * 70)

    scored_matches = score_matches_with_impact(
        matches_df=matches,
        quantity_lbs=forecast["expected_quantity"],
    )

    display_cols = [
        "primary_naics",
        "compatibility_class",
        "match_score",
        "forecasted_quantity_lbs",
        "co2e_avoided_metric_tons",
        "net_economic_impact_usd",
    ]

    print(f"\nImpact estimate based on forecasted {forecast['expected_quantity']:,.2f} lbs of {chemical_name}:\n")
    print(scored_matches[display_cols].to_string(index=False))

    best = scored_matches.iloc[0]

    print()
    print("=" * 70)
    print("RECOMMENDATION SUMMARY")
    print("=" * 70)
    print(f"Best receiving industry (NAICS {int(best['primary_naics'])}):")
    print(f"  Compatibility class: {best['compatibility_class']}")
    print(f"  Match score:         {best['match_score']:.4f}")
    print(f"  Estimated CO2e avoided: {best['co2e_avoided_metric_tons']:,.2f} metric tons")
    print(f"  Estimated net economic impact: ${best['net_economic_impact_usd']:,.2f}")

    return {"forecast": forecast, "matches": scored_matches}


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    # Test case representing a real facility prediction
    run_pipeline(
        facility_state="TX",
        primary_naics_code=325199,
        cas_number="67-56-1",
        chemical_name="Methanol",
        prod_ratio_or_activity="INCREASE",
        latitude=29.7604,
        longitude=-95.3698,
        prod_ratio_activity_index=1.05,
        quantity_prior_year=500000.0, # Added realistic lag
        top_k=5,
        min_records=3,
    )