"""
Build the labelled material-intelligence dataset from the sensor simulator.

Outputs (git-ignored, regenerate with this script)
--------------------------------------------------
data/synthetic/material_sensor_dataset.npz
    X_spectra (N, 256) | X_sensors (N, 8) | y_material (N,) | y_purity (N,)
data/synthetic/material_sensor_meta.json
    channel names, material names, wavelength grid, class counts
data/synthetic/material_sensor_preview.csv
    first 200 rows of scalar sensors + labels for eye-balling
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensor_simulation import (  # noqa: E402
    MATERIAL_NAMES,
    SENSOR_CHANNELS,
    WAVELENGTHS,
    simulate_batch,
)

N_SAMPLES = 9000
SEED = 20240517

OUT_DIR = "data/synthetic"
NPZ_OUT = os.path.join(OUT_DIR, "material_sensor_dataset.npz")
META_OUT = os.path.join(OUT_DIR, "material_sensor_meta.json")
PREVIEW_OUT = os.path.join(OUT_DIR, "material_sensor_preview.csv")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Simulating {N_SAMPLES:,} multi-sensor samples "
          f"across {len(MATERIAL_NAMES)} materials...")

    # mild class imbalance so evaluation is not trivially balanced
    weights = rng.uniform(0.6, 1.4, size=len(MATERIAL_NAMES))
    X_spec, X_sens, y_mat, y_pur = simulate_batch(
        N_SAMPLES, rng, class_weights=weights
    )

    np.savez_compressed(
        NPZ_OUT,
        X_spectra=X_spec.astype(np.float32),
        X_sensors=X_sens.astype(np.float32),
        y_material=y_mat.astype(np.int64),
        y_purity=y_pur.astype(np.float32),
    )

    counts = {MATERIAL_NAMES[i]: int((y_mat == i).sum())
              for i in range(len(MATERIAL_NAMES))}
    meta = {
        "n_samples": int(N_SAMPLES),
        "seed": SEED,
        "sensor_channels": SENSOR_CHANNELS,
        "material_names": MATERIAL_NAMES,
        "wavelengths_nm": [round(float(w), 2) for w in WAVELENGTHS],
        "n_wavelengths": int(len(WAVELENGTHS)),
        "class_counts": counts,
        "purity_summary": {
            "mean": float(y_pur.mean()),
            "p05": float(np.percentile(y_pur, 5)),
            "p50": float(np.percentile(y_pur, 50)),
            "p95": float(np.percentile(y_pur, 95)),
        },
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    preview = pd.DataFrame(X_sens[:200], columns=SENSOR_CHANNELS)
    preview.insert(0, "purity", y_pur[:200].round(4))
    preview.insert(0, "material", [MATERIAL_NAMES[i] for i in y_mat[:200]])
    preview.to_csv(PREVIEW_OUT, index=False)

    print("\nClass counts:")
    for k, v in counts.items():
        print(f"  {k:16s} {v:5d}")
    print(f"\nPurity  mean={y_pur.mean():.3f}  "
          f"p05={np.percentile(y_pur, 5):.3f}  p95={np.percentile(y_pur, 95):.3f}")
    print("\nSaved:")
    print(" ", NPZ_OUT)
    print(" ", META_OUT)
    print(" ", PREVIEW_OUT)


if __name__ == "__main__":
    main()
