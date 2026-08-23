"""Material-to-industry compatibility matching and ranking."""

import pandas as pd

COMPATIBILITY_DATA = (
    "data/processed/material_industry_compatibility.csv"
)


def recommend_industries(
    chemical_name,
    min_records=1,
    top_k=10
):
    """
    Recommend receiving industries for a given waste material.

    Rankings are based on historical compatibility score,
    compatibility confidence, and historical utilization.
    """

    df = pd.read_csv(COMPATIBILITY_DATA)

    # ---------------------------------------------------------
    # Find candidate industries for the requested material
    # ---------------------------------------------------------

    candidates = df[
        df["chemical_name"].str.strip().str.lower()
        == chemical_name.strip().lower()
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Remove extremely weak historical relationships
    candidates = candidates[
        candidates["records"] >= min_records
    ].copy()

    # ---------------------------------------------------------
    # Ranking score
    #
    # Compatibility score is the primary signal.
    # Confidence reflects reliability of the classification.
    # Record count provides a small historical-support signal.
    # ---------------------------------------------------------

    candidates["match_score"] = (
        0.60 * candidates["compatibility_score"]
        + 0.30 * candidates["compatibility_confidence"]
        + 0.10
        * (
            candidates["records"]
            / candidates["records"].max()
        )
    )

    # ---------------------------------------------------------
    # Sort recommendations
    # ---------------------------------------------------------

    candidates = candidates.sort_values(
        "match_score",
        ascending=False
    )

    # ---------------------------------------------------------
    # Return useful fields
    # ---------------------------------------------------------

    result = candidates[
        [
            "chemical_name",
            "primary_naics",
            "records",
            "compatibility_score",
            "compatibility_confidence",
            "compatibility_class",
            "recycling_rate",
            "treatment_rate",
            "energy_recovery_rate",
            "management_rate",
            "match_score",
        ]
    ].head(top_k)

    return result.reset_index(drop=True)


if __name__ == "__main__":

    chemical = "Methanol"

    print("=" * 70)
    print("INDUSTRIAL SYMBIOSIS MATCHING ENGINE")
    print("=" * 70)

    print(f"\nWaste material: {chemical}")

    recommendations = recommend_industries(
        chemical_name=chemical,
        min_records=5,
        top_k=10
    )

    if recommendations.empty:

        print(
            "\nNo compatible receiving industries "
            "were found."
        )

    else:

        print("\nTop receiving industries:\n")

        print(
            recommendations.to_string(
                index=False
            )
        )