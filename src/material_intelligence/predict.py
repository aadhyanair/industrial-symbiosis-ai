"""
Inference helper: raw digital-twin sensor frame -> material + purity.

A "frame" is one observation from the simulator / twin:
    spectrum : (N_WAVELENGTHS,)  raw reflectance
    sensors  : (N_SENSORS,)      raw scalar sensor bank (SENSOR_CHANNELS order)
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnn_model import MultiTaskCNN1D  # noqa: E402
from preprocess import build_spectral_channels  # noqa: E402

TRANSFORM = "models/material_intelligence/preprocess.joblib"
MODEL = "models/material_intelligence/cnn.pt"


class MaterialIntelligence:
    def __init__(self, model_path=MODEL, transform_path=TRANSFORM):
        self.meta = joblib.load(transform_path)
        self.scaler = self.meta["sensor_scaler"]
        self.material_names = self.meta["material_names"]
        self.sensor_channels = self.meta["sensor_channels"]

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.model = MultiTaskCNN1D(cfg["spec_len"], cfg["n_sensors"],
                                    cfg["n_classes"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    # ------------------------------------------------------------------

    def _prep(self, spectra, sensors):
        spectra = np.atleast_2d(np.asarray(spectra, dtype=np.float64))
        sensors = np.atleast_2d(np.asarray(sensors, dtype=np.float64))
        Xc = build_spectral_channels(spectra)                 # (n, 2, L)
        Xs = self.scaler.transform(sensors).astype(np.float32)  # (n, S)
        return Xc.astype(np.float32), Xs

    def predict(self, spectrum, sensors, top_k=3):
        Xc, Xs = self._prep(spectrum, sensors)
        probs, purity = self.model.predict(Xc, Xs)
        out = []
        for p, pur in zip(probs, purity):
            order = np.argsort(p)[::-1][:top_k]
            out.append({
                "material": self.material_names[int(order[0])],
                "material_confidence": float(p[order[0]]),
                "top_k": [
                    {"material": self.material_names[int(j)],
                     "probability": float(p[j])}
                    for j in order
                ],
                "purity": float(pur),
                "purity_pct": round(float(pur) * 100, 2),
            })
        return out[0] if len(out) == 1 else out

    def predict_flat(self, flat_batch):
        """flat_batch: (n, 2*L + S) already-preprocessed - used by SHAP tooling."""
        with torch.no_grad():
            logits, purity = self.model(torch.as_tensor(flat_batch,
                                                        dtype=torch.float32))
        return torch.softmax(logits, 1).numpy(), purity.numpy()


if __name__ == "__main__":
    # round-trip test against a fresh simulated sample
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "digital_twin"))
    from sensor_simulation import simulate_sample  # noqa: E402

    rng = np.random.default_rng(7)
    truth = simulate_sample("PET", rng, purity=0.82)
    mi = MaterialIntelligence()
    pred = mi.predict(truth["spectrum"], truth["sensors"])
    print("true material :", truth["material"], "| true purity:",
          round(truth["purity"], 3))
    print("pred material :", pred["material"],
          f"({pred['material_confidence']:.3f}) | pred purity:",
          round(pred["purity"], 3))
    print("top-k:", pred["top_k"])
