"""
Stateful digital twin of a facility waste / by-product stream.

The twin carries a small physical state (material composition, moisture,
accumulated contamination, sensor drift) and advances it through time.  At any
step it can emit a simulated multi-sensor reading via ``sensor_simulation``.

It is intentionally lightweight - the point is to produce a realistic,
time-correlated sensor stream that the Material-Intelligence models consume,
and to expose a clean hand-off object for the industrial-symbiosis pipeline
(predicted material + purity + estimated quantity).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:  # package import
    from .sensor_simulation import (
        MATERIALS,
        MATERIAL_NAMES,
        SENSOR_CHANNELS,
        N_WAVELENGTHS,
        simulate_sample,
    )
except ImportError:  # run directly as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sensor_simulation import (
        MATERIALS,
        MATERIAL_NAMES,
        SENSOR_CHANNELS,
        N_WAVELENGTHS,
        simulate_sample,
    )


class DigitalTwin:
    """
    Parameters
    ----------
    facility_id : str
    composition : dict[str, float]
        Mass fractions per material (need not sum to 1 - it is normalised).
    throughput_kg_h : float
        Nominal stream mass flow, used for quantity hand-off.
    ambient_humidity : float
        Target moisture (%) the stream relaxes toward.
    handling_intensity : float in [0, 1]
        Higher -> faster contamination build-up and moisture pickup.
    seed : int
    """

    def __init__(self, facility_id, composition, throughput_kg_h=1000.0,
                 ambient_humidity=8.0, handling_intensity=0.3, seed=0):
        self.facility_id = facility_id
        self.rng = np.random.default_rng(seed)

        mats = {m: float(w) for m, w in composition.items() if w > 0}
        for m in mats:
            if m not in MATERIALS:
                raise KeyError(f"unknown material {m!r}")
        total = sum(mats.values())
        self.composition = {m: w / total for m, w in mats.items()}

        self.throughput_kg_h = float(throughput_kg_h)
        self.ambient_humidity = float(ambient_humidity)
        self.handling_intensity = float(np.clip(handling_intensity, 0.0, 1.0))

        self.time_h = 0.0
        self.step_idx = 0

        dominant = max(self.composition, key=self.composition.get)
        self.dominant_material = dominant
        self.moisture_pct = MATERIALS[dominant]["moisture"][0]
        # start fairly clean; contamination is a [0, 1] loss of purity
        self.contamination = float(self.rng.uniform(0.02, 0.12))
        self.sensor_drift = np.zeros(len(SENSOR_CHANNELS))

    # ------------------------------------------------------------------
    # state evolution
    # ------------------------------------------------------------------

    @property
    def purity(self):
        return float(np.clip(1.0 - self.contamination, 0.0, 1.0))

    def step(self, dt_h=1.0):
        """Advance the twin by ``dt_h`` hours."""
        self.step_idx += 1
        self.time_h += dt_h
        h = self.handling_intensity

        # 1. moisture relaxes toward ambient, with noise
        tau = 6.0
        self.moisture_pct += (
            (self.ambient_humidity - self.moisture_pct) * (dt_h / tau)
            + self.rng.normal(0, 0.15) * (1.0 + h)
        )
        self.moisture_pct = max(self.moisture_pct, 0.0)

        # 2. slow contamination build-up from handling / cross-mixing
        self.contamination += self.rng.normal(0.004 * h, 0.002)

        # 3. occasional contamination shock (mis-sort, spill)
        if self.rng.random() < 0.03 + 0.05 * h:
            self.contamination += self.rng.uniform(0.05, 0.25)

        # 4. rare cleaning / re-sort event
        if self.rng.random() < 0.02:
            self.contamination *= self.rng.uniform(0.3, 0.6)

        self.contamination = float(np.clip(self.contamination, 0.0, 0.95))

        # 5. sensor drift random walk
        self.sensor_drift += self.rng.normal(0, 0.01, size=self.sensor_drift.shape)

        # 6. dominant material can shift if composition is close
        if self.rng.random() < 0.05:
            mats = list(self.composition)
            weights = np.array([self.composition[m] for m in mats])
            self.dominant_material = mats[
                int(self.rng.choice(len(mats), p=weights / weights.sum()))
            ]

        return self.read_sensors()

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def read_sensors(self):
        """Emit one simulated sensor frame for the current twin state."""
        others = [m for m in MATERIAL_NAMES if m != self.dominant_material]
        # contaminant biased toward whatever else is in the stream
        in_stream = [m for m in self.composition if m != self.dominant_material]
        if in_stream and self.rng.random() < 0.7:
            contaminant = in_stream[int(self.rng.integers(len(in_stream)))]
        else:
            contaminant = others[int(self.rng.integers(len(others)))]

        sample = simulate_sample(
            self.dominant_material,
            self.rng,
            purity=self.purity,
            moisture_pct=self.moisture_pct,
            contaminant=contaminant,
        )
        sensors = sample["sensors"] + self.sensor_drift
        # channels that cannot be negative: moisture, density, conductivity,
        # voc, chlorine_index, ir_reflectance  (indices 0,1,3,4,5,6)
        for i in (0, 1, 3, 4, 5, 6):
            sensors[i] = max(sensors[i], 0.0)
        sample["sensors"] = sensors
        sample["facility_id"] = self.facility_id
        sample["time_h"] = self.time_h
        sample["step"] = self.step_idx
        return sample

    def run(self, steps, dt_h=1.0):
        """
        Advance ``steps`` times and return a tidy time-series ``DataFrame``
        (scalar sensors + state); the raw spectra are returned separately.
        """
        rows, spectra = [], []
        for _ in range(steps):
            s = self.step(dt_h)
            spectra.append(s["spectrum"])
            row = {
                "facility_id": s["facility_id"],
                "step": s["step"],
                "time_h": s["time_h"],
                "dominant_material": s["material"],
                "contaminant": s["contaminant"],
                "true_purity": s["purity"],
            }
            row.update(dict(zip(SENSOR_CHANNELS, s["sensors"])))
            rows.append(row)
        df = pd.DataFrame(rows)
        spectra = np.vstack(spectra) if spectra else np.empty((0, N_WAVELENGTHS))
        return df, spectra

    # ------------------------------------------------------------------
    # pipeline hand-off
    # ------------------------------------------------------------------

    def handoff(self, predicted_material=None, predicted_purity=None,
                window_h=24.0):
        """
        Structured object for the industrial-symbiosis matching / impact stage.

        ``predicted_*`` come from the Material-Intelligence model; if omitted the
        twin's own (ground-truth) state is used.
        """
        mat = predicted_material or self.dominant_material
        pur = self.purity if predicted_purity is None else float(predicted_purity)
        quantity_kg = self.throughput_kg_h * window_h
        return {
            "facility_id": self.facility_id,
            "material": mat,
            "purity": round(pur, 4),
            "moisture_pct": round(self.moisture_pct, 3),
            "estimated_quantity_kg": round(quantity_kg, 1),
            "estimated_quantity_lbs": round(quantity_kg * 2.20462, 1),
            "reuse_ready": bool(pur >= 0.85 and self.moisture_pct < 15.0),
        }


if __name__ == "__main__":
    twin = DigitalTwin(
        facility_id="FAC-001",
        composition={"HDPE": 0.7, "PP": 0.2, "PET": 0.1},
        throughput_kg_h=1200.0,
        handling_intensity=0.4,
        seed=42,
    )
    df, spectra = twin.run(steps=24)
    print(df[["step", "dominant_material", "true_purity",
              "moisture_pct", "voc_ppm"]].round(3).to_string(index=False))
    print("\nspectra block:", spectra.shape)
    print("\nhand-off:", twin.handoff())
