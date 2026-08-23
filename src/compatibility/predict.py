"""
Prediction utilities for compatibility risk.

Feature list updated to match the leak-free feature set in
preprocess.py (recycling_rate, energy_recovery_rate,
historical_utilization_score, and management_rate removed - see that
module's docstring for why).
"""

import pandas as pd
import joblib

MODEL = "models/compatibility/random_forest.joblib"
PREPROCESSOR = "models/compatibility/preprocessor.joblib"
LABEL_ENCODER = "models/compatibility/label_encoder.joblib"

FEATURES = [
    "chemical_name",
    "primary_naics",
    "records",
    "total_release",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed",
    "treatment_rate",
]


def load_objects():

    model = joblib.load(MODEL)
    preprocessor = joblib.load(PREPROCESSOR)
    label_encoder = joblib.load(LABEL_ENCODER)

    return model, preprocessor, label_encoder


def predict_compatibility(
    chemical_name,
    primary_naics,
    records,
    total_release,
    total_recycled,
    total_treated,
    total_energy_recovery,
    total_managed,
    treatment_rate,
):

    model, preprocessor, label_encoder = load_objects()

    data = pd.DataFrame(
        [{
            "chemical_name": chemical_name,
            "primary_naics": primary_naics,
            "records": records,
            "total_release": total_release,
            "total_recycled": total_recycled,
            "total_treated": total_treated,
            "total_energy_recovery": total_energy_recovery,
            "total_managed": total_managed,
            "treatment_rate": treatment_rate,
        }]
    )

    data = data[FEATURES]

    processed = preprocessor.transform(data)

    # Preserve feature names to avoid the sklearn "missing feature
    # names" warning (model was trained on a named DataFrame).
    try:
        feature_names = preprocessor.get_feature_names_out()
        processed = pd.DataFrame(processed, columns=feature_names)
    except Exception:
        pass

    prediction = model.predict(processed)[0]

    probabilities = model.predict_proba(processed)[0]

    predicted_class = label_encoder.inverse_transform(
        [prediction]
    )[0]

    probability_map = {
        label_encoder.inverse_transform([i])[0]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    confidence = float(max(probabilities))

    return {
        "compatibility_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_map,
    }


if __name__ == "__main__":

    result = predict_compatibility(
        chemical_name="Methanol",
        primary_naics=325199,
        records=227,
        total_release=6645461,
        total_recycled=6645461,
        total_treated=6645461,
        total_energy_recovery=6645461,
        total_managed=493871500,
        treatment_rate=1.0,
    )

    print("\nCompatibility Prediction")
    print("=" * 40)

    print(
        f"Class: {result['compatibility_class']}"
    )

    print(
        f"Confidence: {result['confidence']:.4f}"
    )

    print("\nProbabilities:")

    for cls, probability in result["probabilities"].items():

        print(
            f"{cls}: {probability:.4f}"
        )