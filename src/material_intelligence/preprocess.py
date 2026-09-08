"""
Spectral + sensor preprocessing for the Material-Intelligence models.

Steps
-----
1. Load the simulated dataset (data/synthetic/material_sensor_dataset.npz).
2. Spectral pipeline, per sample:
       * SNV  (standard normal variate) - removes multiplicative scatter
       * Savitzky-Golay 1st derivative  - removes additive baseline, sharpens bands
   The two are stacked as a 2-channel signal for the 1-D CNN: (2, N_WAVELENGTHS).
3. Scalar sensors -> StandardScaler (fit on train only).
4. Stratified train / val / test split (70 / 15 / 15) by material class.
5. Save split arrays to data/processed/ and the fitted transforms to
   models/material_intelligence/preprocess.joblib.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_NPZ = "data/synthetic/material_sensor_dataset.npz"
META_JSON = "data/synthetic/material_sensor_meta.json"

OUT_NPZ = "data/processed/mi_dataset.npz"
TRANSFORM_OUT = "models/material_intelligence/preprocess.joblib"

SG_WINDOW = 11
SG_POLY = 3
RANDOM_STATE = 42


def snv(spectra):
    """Standard normal variate, row-wise."""
    mu = spectra.mean(axis=1, keepdims=True)
    sd = spectra.std(axis=1, keepdims=True)
    return (spectra - mu) / np.where(sd == 0, 1.0, sd)


def sg_first_derivative(spectra):
    return savgol_filter(
        spectra, window_length=SG_WINDOW, polyorder=SG_POLY,
        deriv=1, axis=1, mode="interp",
    )


def build_spectral_channels(spectra):
    """
    (N, L) reflectance -> (N, 2, L) : [SNV, SG-1st-derivative(SNV)].

    The derivative channel is row-normalised by its own mean-absolute value so
    the transform is stateless and identical for a full batch or a single
    sample (important: training and inference must match exactly).
    """
    s = snv(spectra.astype(np.float64))
    d = sg_first_derivative(s)
    d = d / (np.abs(d).mean(axis=1, keepdims=True) + 1e-8)
    return np.stack([s, d], axis=1).astype(np.float32)


def main():
    if not os.path.exists(RAW_NPZ):
        raise FileNotFoundError(
            f"{RAW_NPZ} not found - run "
            "`python src/digital_twin/generate_dataset.py` first."
        )

    data = np.load(RAW_NPZ)
    X_spec = data["X_spectra"]
    X_sens = data["X_sensors"]
    y_mat = data["y_material"]
    y_pur = data["y_purity"]

    with open(META_JSON) as f:
        meta = json.load(f)
    material_names = meta["material_names"]
    sensor_channels = meta["sensor_channels"]

    print(f"Loaded {len(X_spec):,} samples | "
          f"spectra {X_spec.shape} | sensors {X_sens.shape}")

    # -- stratified split: train / (val+test) then val / test -----------
    idx = np.arange(len(X_spec))
    tr, tmp = train_test_split(
        idx, test_size=0.30, random_state=RANDOM_STATE, stratify=y_mat
    )
    va, te = train_test_split(
        tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_mat[tmp]
    )
    print(f"Split -> train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    # -- spectral channels (stateless transform) -----------------------
    Xc = build_spectral_channels(X_spec)

    # -- scalar sensor scaling (fit on train only) --------------------
    scaler = StandardScaler().fit(X_sens[tr])
    Xs = scaler.transform(X_sens).astype(np.float32)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/material_intelligence", exist_ok=True)

    np.savez_compressed(
        OUT_NPZ,
        Xspec_train=Xc[tr], Xsens_train=Xs[tr],
        ymat_train=y_mat[tr], ypur_train=y_pur[tr],
        Xspec_val=Xc[va], Xsens_val=Xs[va],
        ymat_val=y_mat[va], ypur_val=y_pur[va],
        Xspec_test=Xc[te], Xsens_test=Xs[te],
        ymat_test=y_mat[te], ypur_test=y_pur[te],
    )

    joblib.dump(
        {
            "sensor_scaler": scaler,
            "material_names": material_names,
            "sensor_channels": sensor_channels,
            "spectral_channels": ["snv", "sg1_derivative"],
            "wavelengths_nm": meta["wavelengths_nm"],
            "sg_window": SG_WINDOW,
            "sg_poly": SG_POLY,
        },
        TRANSFORM_OUT,
    )

    print("\nSaved:")
    print(" ", OUT_NPZ)
    print(" ", TRANSFORM_OUT)


if __name__ == "__main__":
    main()
