"""
Multi-task 1-D CNN: NIR spectrum + chemical sensors -> (material class, purity).

Architecture
------------
    spectrum (B, 2, L) --3x [Conv1d-BN-ReLU-Pool] --> GAP (B, 64)
    sensors  (B, S)     --------------- MLP --------> (B, 16)
                         concat -> Dense(64)-ReLU-Dropout
                             |-- class head  -> logits (B, n_classes)
                             |-- purity head -> sigmoid (B, 1)

The whole network also accepts a single **flat** tensor
``x = [spectrum.flatten(1) | sensors]`` of width ``2*L + S`` so that SHAP
gradient/deep explainers (which expect one input tensor and one output tensor)
can attribute over every wavelength and sensor channel.  ``SingleHead`` wraps the
model to expose exactly one scalar/vector output for the explainer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MultiTaskCNN1D(nn.Module):
    def __init__(self, spec_len, n_sensors, n_classes,
                 spec_channels=2, dropout=0.25):
        super().__init__()
        self.spec_len = int(spec_len)
        self.spec_channels = int(spec_channels)
        self.n_sensors = int(n_sensors)
        self.n_classes = int(n_classes)

        def block(cin, cout, k):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        self.conv = nn.Sequential(
            block(spec_channels, 16, 9),
            block(16, 32, 7),
            block(32, 64, 5),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.sensor_mlp = nn.Sequential(
            nn.Linear(n_sensors, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 16, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(64, n_classes)
        self.purity_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    # ------------------------------------------------------------------
    # flat <-> (spectrum, sensors) helpers
    # ------------------------------------------------------------------

    @property
    def flat_dim(self):
        return self.spec_channels * self.spec_len + self.n_sensors

    def unpack(self, x):
        spec = x[:, : self.spec_channels * self.spec_len]
        spec = spec.view(-1, self.spec_channels, self.spec_len)
        sens = x[:, self.spec_channels * self.spec_len:]
        return spec, sens

    @staticmethod
    def pack(spec, sens):
        if isinstance(spec, np.ndarray):
            spec = torch.as_tensor(spec, dtype=torch.float32)
        if isinstance(sens, np.ndarray):
            sens = torch.as_tensor(sens, dtype=torch.float32)
        return torch.cat([spec.flatten(1), sens], dim=1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def features(self, spec, sens):
        h = self.conv(spec)
        h = self.gap(h).squeeze(-1)
        s = self.sensor_mlp(sens)
        return self.fusion(torch.cat([h, s], dim=1))

    def forward(self, x, sens=None):
        """
        Two call styles:
          forward(spec (B,2,L), sens (B,S))  -> (logits, purity)
          forward(flat (B, 2L+S))            -> (logits, purity)
        """
        if sens is None:
            spec, sens = self.unpack(x)
        else:
            spec = x
        z = self.features(spec, sens)
        return self.class_head(z), self.purity_head(z).squeeze(-1)

    # ------------------------------------------------------------------
    # convenience inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, spec, sens):
        self.eval()
        flat = self.pack(spec, sens)
        logits, purity = self.forward(flat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs, purity.cpu().numpy()


class SingleHead(nn.Module):
    """
    Wrap ``MultiTaskCNN1D`` to return a single tensor for SHAP.

    which="class"  -> class logits           (B, n_classes)
    which="purity" -> purity                 (B, 1)
    """

    def __init__(self, model: MultiTaskCNN1D, which="class"):
        super().__init__()
        self.model = model
        self.which = which

    def forward(self, x):
        logits, purity = self.model(x)
        if self.which == "purity":
            return purity.unsqueeze(-1)
        return logits


def build_model(meta_or_dims):
    """Instantiate from a preprocess.joblib dict or an explicit dims dict."""
    d = meta_or_dims
    spec_len = len(d["wavelengths_nm"]) if "wavelengths_nm" in d else d["spec_len"]
    n_sensors = len(d["sensor_channels"]) if "sensor_channels" in d else d["n_sensors"]
    n_classes = len(d["material_names"]) if "material_names" in d else d["n_classes"]
    return MultiTaskCNN1D(spec_len=spec_len, n_sensors=n_sensors,
                          n_classes=n_classes)


if __name__ == "__main__":
    m = MultiTaskCNN1D(spec_len=256, n_sensors=8, n_classes=15)
    n_params = sum(p.numel() for p in m.parameters())
    print(m)
    print(f"\ntrainable parameters: {n_params:,}")
    dummy = torch.randn(4, m.flat_dim)
    lg, pu = m(dummy)
    print("logits:", tuple(lg.shape), "purity:", tuple(pu.shape))
