# Person B — Digital Twin + Material Intelligence

This half of the project sits *upstream* of Person A's forecast → match → impact
pipeline. It answers: **given a physical waste stream at a facility, what material
is it and how pure/contaminated is it?** — using a simulated sensor rig and a
1-D CNN, with SHAP explanations for every call.

```
DigitalTwin (waste stream)
    │  simulated NIR spectrum + moisture + chemical sensors, stepped through time
    ▼
Latest multi-sensor frame
    │
    ▼
1-D CNN  ──►  material class  (15 classes)
         └─►  purity estimate (continuous 0–1)
    │
    ▼
SHAP  ──►  which wavelengths / sensors drove the prediction
    │
    ▼
Structured hand-off  ──►  Person A: recommend_industries() + impact
```

## Components

| Area | File | What it does |
|---|---|---|
| Sensor simulation | [`src/digital_twin/sensor_simulation.py`](src/digital_twin/sensor_simulation.py) | Physics-style NIR reflectance (1000–2500 nm, 256 bands) built from material-characteristic C-H / O-H / C-Cl overtone bands; moisture injects water bands; 8 scalar chemical sensors (density, pH, conductivity, VOC, chlorine index, IR reflectance, temperature). Contamination = convex blend of target + random contaminant material → continuous purity label. |
| Digital twin | [`src/digital_twin/digital_twin.py`](src/digital_twin/digital_twin.py) | `DigitalTwin` class holding stream state (composition, moisture, accumulated contamination, sensor drift). `.step()` evolves it (moisture relaxes to ambient, contamination builds up, random shock / cleaning events); `.run(n)` returns a time series; `.handoff()` emits the object Person A consumes. |
| Dataset build | [`src/digital_twin/generate_dataset.py`](src/digital_twin/generate_dataset.py) | Simulates 9 000 labelled samples → `data/synthetic/material_sensor_dataset.npz` (+ meta JSON + preview CSV). |
| Preprocessing | [`src/material_intelligence/preprocess.py`](src/material_intelligence/preprocess.py) | SNV + Savitzky–Golay 1st-derivative → 2-channel spectral signal; `StandardScaler` on sensors (fit on train only); stratified 70/15/15 split. |
| 1-D CNN | [`src/material_intelligence/cnn_model.py`](src/material_intelligence/cnn_model.py) | `MultiTaskCNN1D` (PyTorch): 3 Conv1d-BN-ReLU-Pool blocks on the spectrum → global average pool, small MLP on the sensors, fused dense layer → **classification head** (cross-entropy) + **purity head** (sigmoid, Huber loss). Also accepts a single flat tensor so SHAP can attribute over all inputs. |
| Training | [`src/material_intelligence/train.py`](src/material_intelligence/train.py) | Trains the CNN (Adam, class-weighted CE + 6·Huber, early stopping), benchmarks against a RandomForest baseline, writes metrics + plots. |
| Inference | [`src/material_intelligence/predict.py`](src/material_intelligence/predict.py) | `MaterialIntelligence.predict(spectrum, sensors)` → material, top-k probs, purity. |
| Explainability | [`src/material_intelligence/explain_shap.py`](src/material_intelligence/explain_shap.py) | `shap.GradientExplainer` on **both** heads; aggregates mean\|SHAP\| into per-wavelength curves, per-sensor bars, group mass fractions (SNV / derivative / sensors), per-class sensor importance, and one local example. |
| End-to-end demo | [`src/material_intelligence_pipeline.py`](src/material_intelligence_pipeline.py) | Twin → CNN → SHAP drivers → hand-off, with an optional bridge into Person A's `recommend_industries()`. |

## How to run

```bash
py -3.13 src/digital_twin/generate_dataset.py
py -3.13 src/material_intelligence/preprocess.py
py -3.13 src/material_intelligence/train.py
py -3.13 src/material_intelligence/explain_shap.py
py -3.13 src/material_intelligence_pipeline.py
```

Dependencies (Python 3.13): `numpy scipy pandas scikit-learn joblib matplotlib
torch shap`.

## Results

_Test set: 1 350 samples, 15 material classes._

| Model | Material accuracy | Material macro-F1 | Purity MAE | Purity R² |
|---|---|---|---|---|
| **Multi-task 1-D CNN** (~21.5k params, early-stopped ~epoch 60) | **0.939** | **0.936** | **0.057** | **0.695** |
| RandomForest baseline (flattened features) | 0.938 | 0.934 | 0.049 | 0.718 |

The CNN and the RandomForest are close on classification; the RandomForest is
marginally better on purity regression on this (clean, synthetic) data. Both are
reported so the comparison is honest. The value of the CNN here is the shared
spectral backbone + single explainable model for both tasks.

Full metrics + confusion matrix: [`results/material_intelligence/cnn_results.json`](results/material_intelligence/cnn_results.json).
Plots: `training_curve.png`, `confusion_matrix.png`, `purity_scatter.png`,
`shap_spectral_importance.png`, `shap_sensor_importance.png` in the same folder.

### SHAP attribution (both heads, `GradientExplainer`, 300 test samples)

Attribution mass splits as **spectrum ≈ 93 % (material) / ≈ 98 % (purity),
chemical sensors ≈ 7 % / 2 %**, with the Savitzky–Golay derivative channel
carrying slightly more than the raw SNV channel — i.e. the CNN keys on band
*shape*, as expected for NIR.

| Head | Top wavelengths | Top sensors |
|---|---|---|
| Material | ~1650–1710 nm (C–H overtone fingerprint) | density, IR reflectance, chlorine index |
| Purity | ~1870 nm & ~2170–2190 nm (O–H water / C–H combination) | IR reflectance, density, moisture, VOC |

Two useful sanity checks: (1) the sensor rankings match the simulator design —
metals ⇒ high density / IR reflectance / conductivity, PVC ⇒ chlorine index; and
(2) the two heads look at *different* spectral regions — the material head at the
C–H fingerprint, the purity head at the water / contaminant-combination bands —
which is exactly the physical story for "how contaminated is this?".

Files: `shap_material.json`, `shap_purity.json`, `shap_local_example.json`,
`shap_spectral_importance.png`, `shap_sensor_importance.png`.

## Notes / honesty

* All sensor data is **synthetic** — a physically-motivated simulator, not real
  spectrometer readings. Band positions follow known NIR overtone regions but
  amplitudes are illustrative and chosen to keep the 15 classes separable.
* Purity is defined by the simulator as `1 − contaminant_blend_weight`, so the
  regression target is exact by construction; real purity labels would be
  noisier.
* The digital twin is a lightweight state model (a few ODE-style relaxations +
  stochastic events), enough to produce time-correlated streams for the models
  to consume.
