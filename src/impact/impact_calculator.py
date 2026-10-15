"""
Impact Estimation Module (Functionality #4)
Calculates environmental and economic benefits of industrial symbiosis matches.

Uses illustrative WARM-style factors to estimate CO2e avoided and economic savings.
"""

import pandas as pd

LBS_PER_TON = 2000.0

# Illustrative Emission Factors (MTCO2e avoided per short ton diverted)
# Relative to a landfill/disposal baseline of 0.
EMISSION_FACTOR_RECYCLING = 1.50
EMISSION_FACTOR_ENERGY_RECOVERY = 0.60
EMISSION_FACTOR_TREATMENT = 0.10
EMISSION_FACTOR_DISPOSAL = 0.0

# Illustrative Economic Factors (USD per short ton)
DISPOSAL_COST_PER_TON = 50.0        # Avoided cost of landfill/disposal
RECYCLING_REVENUE_PER_TON = 30.0    # Recovered-material revenue
TREATMENT_COST_PER_TON = 20.0       # Cost to treat (negative impact on net)
CARBON_VALUE_USD_PER_MTCO2E = 51.0  # Social cost of carbon


def calculate_impact(
    quantity_lbs, 
    recycling_rate=0.0, 
    treatment_rate=0.0, 
    energy_recovery_rate=0.0
):
    """
    Estimates the CO2 and economic savings of a waste exchange based on 
    the receiver's historical management profile.
    """
    tons = quantity_lbs / LBS_PER_TON

    # Ensure rates are non-negative
    recycling_rate = max(float(recycling_rate), 0.0)
    treatment_rate = max(float(treatment_rate), 0.0)
    energy_recovery_rate = max(float(energy_recovery_rate), 0.0)

    # Normalise pathway shares if the sum exceeds 1.0 (100% of material)
    total_pathways = recycling_rate + treatment_rate + energy_recovery_rate
    if total_pathways > 1.0:
        recycling_rate /= total_pathways
        treatment_rate /= total_pathways
        energy_recovery_rate /= total_pathways

    # 1. Environmental Impact
    co2e_avoided = tons * (
        (recycling_rate * EMISSION_FACTOR_RECYCLING) +
        (treatment_rate * EMISSION_FACTOR_TREATMENT) +
        (energy_recovery_rate * EMISSION_FACTOR_ENERGY_RECOVERY)
    )

    # 2. Economic Impact
    # Any managed waste (recycled, treated, energy recovered) avoids disposal costs
    managed_rate = min(recycling_rate + treatment_rate + energy_recovery_rate, 1.0)
    avoided_disposal_cost = tons * managed_rate * DISPOSAL_COST_PER_TON
    
    recycling_revenue = tons * recycling_rate * RECYCLING_REVENUE_PER_TON
    treatment_cost = tons * treatment_rate * TREATMENT_COST_PER_TON
    
    net_economic_impact = avoided_disposal_cost + recycling_revenue - treatment_cost

    return {
        "input_quantity_lbs": quantity_lbs,
        "tons_processed": tons,
        "co2e_avoided_metric_tons": co2e_avoided,
        "co2e_value_usd": co2e_avoided * CARBON_VALUE_USD_PER_MTCO2E,
        "avoided_disposal_cost_usd": avoided_disposal_cost,
        "recycling_revenue_usd": recycling_revenue,
        "treatment_cost_usd": treatment_cost,
        "net_economic_impact_usd": net_economic_impact,
    }


def score_matches_with_impact(matches_df, quantity_lbs):
    """
    Applies the impact calculations to a DataFrame of matched industries.
    """
    if matches_df.empty:
        return matches_df

    records = []
    for _, row in matches_df.iterrows():
        impact = calculate_impact(
            quantity_lbs=quantity_lbs,
            recycling_rate=row.get("recycling_rate", 0.0),
            treatment_rate=row.get("treatment_rate", 0.0),
            energy_recovery_rate=row.get("energy_recovery_rate", 0.0),
        )
        
        record = row.to_dict()
        record["forecasted_quantity_lbs"] = quantity_lbs
        record["co2e_avoided_metric_tons"] = impact["co2e_avoided_metric_tons"]
        record["net_economic_impact_usd"] = impact["net_economic_impact_usd"]
        records.append(record)

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Quick terminal test
    print("=" * 60)
    print("IMPACT ESTIMATION DASHBOARD - TEST RUN")
    print("=" * 60)
    
    test_qty = 50000.0
    result = calculate_impact(
        quantity_lbs=test_qty, 
        recycling_rate=0.70, 
        treatment_rate=0.10, 
        energy_recovery_rate=0.05
    )
    
    print(f"Input Quantity:         {result['input_quantity_lbs']:,.2f} lbs")
    print(f"CO2e Avoided:           {result['co2e_avoided_metric_tons']:,.2f} metric tons")
    print(f"Net Economic Benefit:   ${result['net_economic_impact_usd']:,.2f}")
    print("=" * 60)