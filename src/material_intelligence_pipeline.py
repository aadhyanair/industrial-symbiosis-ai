"""
Person B end-to-end demo: Digital Twin -> Material Intelligence -> hand-off.

    Facility waste stream (DigitalTwin)
            |  simulated NIR + moisture + chemical sensors over time
            v
    Latest multi-sensor frame
            |
            v
    1-D CNN  ->  material class  +  purity estimate
            |
            v
    SHAP  ->  which sensors / wavelengths drove this call
            |
            v
    Structured hand-off  ->  industrial-symbiosis matching + impact (Person A)

Run from the project root (after training the model):

    python src/digital_twin/generate_dataset.py
    python src/material_intelligence/preprocess.py
    python src/material_intelligence/train.py
    python src/material_intelligence/explain_shap.py     # optional, for the report
    python src/material_intelligence_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from digital_twin.digital_twin import DigitalTwin              # noqa: E402
from digital_twin.sensor_simulation import SENSOR_CHANNELS     # noqa: E402
from material_intelligence.predict import MaterialIntelligence  # noqa: E402

# material -> representative TRI chemical, only used for the optional
# Person-A matching bridge (their compatibility table is keyed by TRI name).
MATERIAL_TO_TRI_CHEMICAL = {
    "SOLVENT_WASTE": "Methanol",
    "ORGANIC": "Ethylene glycol",
    "PVC": "Vinyl chloride",
    "PS": "Styrene",
}

PURITY_REUSE_THRESHOLD = 0.85
SHAP_MATERIAL_JSON = "results/material_intelligence/shap_material.json"
SHAP_PURITY_JSON = "results/material_intelligence/shap_purity.json"


def _load_shap_drivers():
    out = {}
    for key, path in (("material", SHAP_MATERIAL_JSON),
                      ("purity", SHAP_PURITY_JSON)):
        if os.path.exists(path):
            with open(path) as f:
                s = json.load(f)
            out[key] = {
                "top_sensors": list(s["sensor_importance"])[:3],
                "top_wavelengths_nm": [
                    round(w["wavelength_nm"]) for w in s["top_wavelengths"][:3]
                ],
                "group_mass_fraction": s["group_mass_fraction"],
            }
    return out


def run(facility_id="FAC-DEMO", composition=None, throughput_kg_h=1400.0,
        steps=24, seed=7):
    composition = composition or {"HDPE": 0.6, "PP": 0.25, "PET": 0.15}

    print("=" * 70)
    print("STEP 1 - DIGITAL TWIN: simulate the waste stream")
    print("=" * 70)
    twin = DigitalTwin(
        facility_id=facility_id,
        composition=composition,
        throughput_kg_h=throughput_kg_h,
        handling_intensity=0.45,
        seed=seed,
    )
    df, spectra = twin.run(steps=steps)
    print(df[["step", "dominant_material", "true_purity", "moisture_pct",
             "voc_ppm", "chlorine_index"]].round(3).tail(8).to_string(index=False))

    latest_spectrum = spectra[-1]
    latest_sensors = df.iloc[-1][list(SENSOR_CHANNELS)].to_numpy(dtype=float)
    true_material = df.iloc[-1]["dominant_material"]
    true_purity = float(df.iloc[-1]["true_purity"])

    print()
    print("=" * 70)
    print("STEP 2 - MATERIAL INTELLIGENCE: 1-D CNN classification + purity")
    print("=" * 70)
    try:
        mi = MaterialIntelligence()
    except FileNotFoundError:
        print("Trained model not found. Run material_intelligence/train.py first.")
        return None

    pred = mi.predict(latest_spectrum, latest_sensors)
    print(f"True (twin state)   : {true_material}  purity={true_purity:.3f}")
    print(f"Predicted material  : {pred['material']} "
          f"(confidence {pred['material_confidence']:.3f})")
    print(f"Predicted purity    : {pred['purity']:.3f} "
          f"({pred['purity_pct']:.1f}%)")
    print("Top-3 material probs :",
          ", ".join(f"{t['material']}={t['probability']:.3f}"
                    for t in pred["top_k"]))

    print()
    print("=" * 70)
    print("STEP 3 - EXPLAINABILITY: SHAP drivers (from explain_shap.py)")
    print("=" * 70)
    drivers = _load_shap_drivers()
    if drivers:
        for head, info in drivers.items():
            print(f"[{head} head] top sensors={info['top_sensors']} | "
                  f"top wavelengths={info['top_wavelengths_nm']} nm | "
                  f"sensor mass={info['group_mass_fraction']['chemical_sensors']:.2f}")
    else:
        print("(run material_intelligence/explain_shap.py to populate SHAP "
              "attribution)")

    print()
    print("=" * 70)
    print("STEP 4 - HAND-OFF to industrial-symbiosis matching / impact")
    print("=" * 70)
    handoff = twin.handoff(
        predicted_material=pred["material"],
        predicted_purity=pred["purity"],
        window_h=float(steps),
    )
    handoff["material_confidence"] = round(pred["material_confidence"], 4)
    handoff["reuse_ready"] = bool(pred["purity"] >= PURITY_REUSE_THRESHOLD
                                  and handoff["moisture_pct"] < 15.0)
    print(json.dumps(handoff, indent=2))

    if not handoff["reuse_ready"]:
        print(f"\n-> purity {pred['purity']:.2f} below reuse threshold "
              f"{PURITY_REUSE_THRESHOLD:.2f}: route to reprocessing / cleaning "
              f"before exchange.")
    else:
        _optional_person_a_bridge(pred["material"], handoff)

    return {"twin_timeseries": df, "prediction": pred, "handoff": handoff,
            "shap_drivers": drivers}


def _optional_person_a_bridge(material, handoff):
    """If Person A's compatibility table exists, show candidate receivers."""
    csv = "data/processed/material_industry_compatibility.csv"
    chem = MATERIAL_TO_TRI_CHEMICAL.get(material)
    if not (os.path.exists(csv) and chem):
        print(f"\n-> reuse-ready. Hand-off object is ready for Person A's "
              f"recommend_industries()/impact stage.")
        return
    try:
        sys.path.insert(0, os.path.join(HERE, "matching"))
        from matching.match_scorer import recommend_industries  # noqa: E402
        print(f"\n-> reuse-ready. Querying Person A matching with proxy "
              f"chemical '{chem}' for {material} ...")
        rec = recommend_industries(chemical_name=chem, min_records=3, top_k=5)
        if rec.empty:
            print("   (no historical compatibility rows for this proxy)")
        else:
            print(rec[["primary_naics", "records", "compatibility_class",
                       "match_score"]].to_string(index=False))
    except Exception as e:  # pragma: no cover
        print(f"   Person A bridge unavailable: {e}")


if __name__ == "__main__":
    run()
