# Purity-gated industrial waste exchange

A pipeline that takes a waste stream from a sensor reading to a ranked list of buyers. Course project for BCSE306L Artificial Intelligence.

In industrial symbiosis, one plant's by-product becomes another plant's raw material. Exchanges often fall through because the receiver cannot tell how clean the material is or how much will arrive. This project measures the stream first, holds it back if it is not clean enough, and only then forecasts the quantity, ranks the industries that can use it and prices the options.

```
discharge point -> NIR spectrum + 8 probes -> 1-D CNN (material class + purity) -> purity gate
                                                                                      |
                              pass: forecast quantity -> rank receiving industries -> CO2e and cost
                              fail: hold for reprocessing
```

## Team

| Person | Module | Folders |
|---|---|---|
| A (Aadhya) | TRI data, forecasting, compatibility, matching, impact, end-to-end pipeline | `src/forecasting`, `src/compatibility`, `src/matching`, `src/impact`, `src/pipeline.py` and the older scripts in `src/` |
| B (Chelintha) | Digital twin, sensor simulation, 1-D CNN, purity prediction, SHAP | `src/digital_twin`, `src/material_intelligence`, `src/material_intelligence_pipeline.py` |
| C (Archisha) | CTGAN, domain adaptation, DiCE | `src/synthetic_generation`, `src/domain_adaptation`, `src/explainability` |

## Setup

Python 3.13. On the development laptop the packages were installed for 3.13 only, so the commands below use `py -3.13`.

```
py -3.13 -m pip install numpy pandas scipy scikit-learn joblib matplotlib torch shap ctgan dice-ml
```

Run every command from the repository root.

## Data and models

`data/raw/`, `data/processed/` and `models/` are git-ignored, so a fresh clone needs the TRI files before the Person A steps will run. Put these in `data/raw/`:

- `US_2A_2021.txt` and `US_2a_2022.txt` for the multi-year forecaster (US EPA Toxics Release Inventory, Basic Plus files)
- `tri_waste_management_2023_corrected.csv` only if you want to rebuild the cleaned single-year table with `src/preprocess.py`

Person B needs no external data. The digital twin generates its own.

## Running it

Person A: forecast, match, impact.

```
py -3.13 src/compatibility/evaluate_and_save.py
py -3.13 src/matching/match_scorer.py
py -3.13 src/impact/impact_calculator.py
py -3.13 src/forecasting/build_multi_year_panel.py
py -3.13 src/forecasting/train_multi_year.py
py -3.13 src/pipeline.py
```

Person B: digital twin and material intelligence. Training takes about two minutes and the SHAP step about five.

```
py -3.13 src/digital_twin/generate_dataset.py
py -3.13 src/material_intelligence/preprocess.py
py -3.13 src/material_intelligence/train.py
py -3.13 src/material_intelligence/explain_shap.py
py -3.13 src/material_intelligence_pipeline.py
```

Person C: extensions on the compatibility table. They read `data/processed/material_industry_compatibility.csv` and the models in `models/compatibility/`.

```
py -3.13 src/synthetic_generation/train_ctgan_v5.py
py -3.13 src/synthetic_generation/evaluate_ctgan_v5.py
py -3.13 src/domain_adaptation/train_domain_adaptation.py
py -3.13 src/domain_adaptation/evaluate_domain_adaptation.py
py -3.13 src/explainability/dice_explain.py
```

`src/synthetic_generation/evaluate_ctgan_distribution.py` reads an older output file that no longer exists and does not run.

## Results

Saved results are in `results/`. Test sets are held out from training.

| Part | Result |
|---|---|
| Material classification (simulated spectra, 1,350 test samples, 15 classes) | CNN 93.93% accuracy, macro-F1 0.936. Random forest 93.78% |
| Purity prediction | CNN MAE 0.057, R2 0.69. Random forest MAE 0.049, R2 0.72 |
| Release forecast, single year (held-out facilities) | Stage 1 AUC 0.88, log-scale R2 0.49 |
| Release forecast, multi-year (2021 to 2022, positive releases) | log-scale R2 0.78, no clear gain over repeating last year's value |
| Methanol example (Texas, 500,000 lb reported) | forecast 234,277 lb, best receiver NAICS 325199 (score 0.98), 85.9 t CO2e avoided |
| CTGAN synthetic profiles | KS distance 0.05 to 0.10 on quantities, 0.19 to 0.54 on rates |
| Domain-adaptation encoder | 91.6% accuracy on 1,969 held-out real pairs |

Raw-pound R2 for the forecaster is close to zero because a few very large releases dominate it, so the log scale is used.

## Limits

- The spectra are simulated. Results for the sensing half say nothing yet about measured streams.
- Only four of the 15 material classes map to a TRI chemical, so only those get ranked receivers. They are solvent waste (methanol), organic waste (ethylene glycol), PVC (vinyl chloride) and polystyrene (styrene).
- The Person B pipeline and the Person A pipeline are separate scripts and are not yet merged into one call.
- The impact factors are illustrative and are not certified life-cycle values.
- The compatibility classifier's 97% accuracy mostly reflects that the band is a formula of its own inputs. The ranking does not use that classifier.

