"""
SHAP attribution for the multi-task 1-D CNN.

Explains BOTH heads over the full input (every NIR wavelength + every chemical
sensor channel):

    * material-classification head : attribution toward each sample's predicted
      material class,
    * purity-regression head       : attribution toward the predicted purity.

Uses ``shap.GradientExplainer`` on the flat input representation
``x = [SNV spectrum (L) | SG-derivative spectrum (L) | sensors (S)]``.
Falls back to ``shap.KernelExplainer`` (model-agnostic) if the gradient
explainer is unavailable.

Outputs
-------
results/material_intelligence/shap_material.json
results/material_intelligence/shap_purity.json
results/material_intelligence/shap_local_example.json
results/material_intelligence/shap_spectral_importance.png
results/material_intelligence/shap_sensor_importance.png
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402
import shap  # noqa: E402

from cnn_model import MultiTaskCNN1D, SingleHead  # noqa: E402

DATA_NPZ = "data/processed/mi_dataset.npz"
TRANSFORM = "models/material_intelligence/preprocess.joblib"
MODEL = "models/material_intelligence/cnn.pt"
OUT_DIR = "results/material_intelligence"

N_BACKGROUND = 128
N_EXPLAIN = 300
SEED = 42


def to_stack(shap_values):
    """Normalise shap output to ndarray (n_samples, n_features, n_outputs)."""
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=-1)
    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return arr[:, :, None]
    return arr


def summarise(abs_shap, spec_len, sensor_channels, wavelengths):
    """abs_shap: (n_features,) mean|shap|. Split into groups + rank."""
    snv = abs_shap[:spec_len]
    deriv = abs_shap[spec_len:2 * spec_len]
    sensors = abs_shap[2 * spec_len:]

    per_wl = snv + deriv
    order = np.argsort(per_wl)[::-1][:15]
    top_wl = [{"wavelength_nm": float(wavelengths[i]),
               "importance": float(per_wl[i])} for i in order]

    sensor_imp = {name: float(v) for name, v in zip(sensor_channels, sensors)}
    sensor_imp = dict(sorted(sensor_imp.items(), key=lambda kv: kv[1],
                             reverse=True))

    total = float(abs_shap.sum()) + 1e-12
    return {
        "group_mass_fraction": {
            "spectrum_snv": float(snv.sum() / total),
            "spectrum_derivative": float(deriv.sum() / total),
            "chemical_sensors": float(sensors.sum() / total),
        },
        "sensor_importance": sensor_imp,
        "top_wavelengths": top_wl,
        "per_wavelength_importance": [float(v) for v in per_wl],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    meta = joblib.load(TRANSFORM)
    material_names = meta["material_names"]
    sensor_channels = meta["sensor_channels"]
    wavelengths = np.asarray(meta["wavelengths_nm"], dtype=float)

    ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = MultiTaskCNN1D(cfg["spec_len"], cfg["n_sensors"], cfg["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    spec_len = cfg["spec_len"]

    d = np.load(DATA_NPZ)
    Xspec_tr, Xsens_tr = d["Xspec_train"], d["Xsens_train"]
    Xspec_te, Xsens_te = d["Xspec_test"], d["Xsens_test"]
    ymat_te, ypur_te = d["ymat_test"], d["ypur_test"]

    def flatten(Xspec, Xsens):
        return np.concatenate(
            [Xspec.reshape(len(Xspec), -1), Xsens], axis=1
        ).astype(np.float32)

    flat_tr = flatten(Xspec_tr, Xsens_tr)
    flat_te = flatten(Xspec_te, Xsens_te)

    bg_idx = rng.choice(len(flat_tr), size=min(N_BACKGROUND, len(flat_tr)),
                        replace=False)
    ex_idx = rng.choice(len(flat_te), size=min(N_EXPLAIN, len(flat_te)),
                        replace=False)
    background = torch.tensor(flat_tr[bg_idx])
    X_explain = torch.tensor(flat_te[ex_idx])

    # predicted classes for the explained samples
    with torch.no_grad():
        logits, purity_pred = model(X_explain)
    pred_cls = logits.argmax(1).numpy()

    feature_names = (
        [f"snv_{w:.0f}nm" for w in wavelengths]
        + [f"deriv_{w:.0f}nm" for w in wavelengths]
        + list(sensor_channels)
    )

    # ---------------- MATERIAL head --------------------------------
    print("SHAP: material-classification head ...")
    try:
        expl_c = shap.GradientExplainer(SingleHead(model, "class"), background)
        sv_c = to_stack(expl_c.shap_values(X_explain))         # (n, F, C)
        method = "GradientExplainer"
    except Exception as e:  # pragma: no cover
        print(f"  gradient explainer failed ({e}); using KernelExplainer")
        f = lambda z: model(torch.tensor(z, dtype=torch.float32))[0].detach().numpy()
        expl_c = shap.KernelExplainer(f, shap.sample(flat_tr[bg_idx], 50))
        sv_c = to_stack(expl_c.shap_values(flat_te[ex_idx], nsamples=200))
        method = "KernelExplainer"

    # attribution toward each sample's predicted class
    n = sv_c.shape[0]
    sv_c_pred = np.stack([sv_c[i, :, pred_cls[i]] for i in range(n)], axis=0)
    abs_c = np.abs(sv_c_pred).mean(axis=0)
    mat_summary = summarise(abs_c, spec_len, sensor_channels, wavelengths)
    mat_summary["explainer"] = method
    mat_summary["n_explained"] = int(n)

    # per-class sensor importance (mean |shap| over samples of that class)
    per_class_sensor = {}
    for c, name in enumerate(material_names):
        mask = pred_cls == c
        if mask.sum() == 0:
            continue
        s = np.abs(sv_c[mask][:, 2 * spec_len:, c]).mean(axis=0)
        per_class_sensor[name] = {
            sc: float(v) for sc, v in zip(sensor_channels, s)
        }
    mat_summary["per_class_sensor_importance"] = per_class_sensor

    with open(os.path.join(OUT_DIR, "shap_material.json"), "w") as fh:
        json.dump(mat_summary, fh, indent=2)

    # ---------------- PURITY head --------------------------------
    print("SHAP: purity-regression head ...")
    try:
        expl_p = shap.GradientExplainer(SingleHead(model, "purity"), background)
        sv_p = to_stack(expl_p.shap_values(X_explain))[:, :, 0]   # (n, F)
    except Exception as e:  # pragma: no cover
        print(f"  gradient explainer failed ({e}); using KernelExplainer")
        f = lambda z: model(torch.tensor(z, dtype=torch.float32))[1].detach().numpy()
        expl_p = shap.KernelExplainer(f, shap.sample(flat_tr[bg_idx], 50))
        sv_p = to_stack(expl_p.shap_values(flat_te[ex_idx], nsamples=200))[:, :, 0]

    abs_p = np.abs(sv_p).mean(axis=0)
    pur_summary = summarise(abs_p, spec_len, sensor_channels, wavelengths)
    pur_summary["explainer"] = method
    pur_summary["n_explained"] = int(sv_p.shape[0])
    with open(os.path.join(OUT_DIR, "shap_purity.json"), "w") as fh:
        json.dump(pur_summary, fh, indent=2)

    # ---------------- local example ------------------------------
    j = 0
    local = {
        "true_material": material_names[int(ymat_te[ex_idx[j]])],
        "predicted_material": material_names[int(pred_cls[j])],
        "true_purity": float(ypur_te[ex_idx[j]]),
        "predicted_purity": float(purity_pred[j]),
        "material_head_top_features": _top_named(sv_c_pred[j], feature_names, 8),
        "purity_head_top_features": _top_named(sv_p[j], feature_names, 8),
    }
    with open(os.path.join(OUT_DIR, "shap_local_example.json"), "w") as fh:
        json.dump(local, fh, indent=2)

    # ---------------- plots -------------------------------------
    per_wl_c = np.asarray(mat_summary["per_wavelength_importance"])
    per_wl_p = np.asarray(pur_summary["per_wavelength_importance"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavelengths, per_wl_c / per_wl_c.max(), label="material head")
    ax.plot(wavelengths, per_wl_p / per_wl_p.max(), label="purity head")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("normalised mean |SHAP|")
    ax.set_title("Spectral attribution by head")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "shap_spectral_importance.png"), dpi=120)
    plt.close(fig)

    s_c = np.array([mat_summary["sensor_importance"][k] for k in sensor_channels])
    s_p = np.array([pur_summary["sensor_importance"][k] for k in sensor_channels])
    x = np.arange(len(sensor_channels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, s_c / s_c.sum(), width=0.4, label="material head")
    ax.bar(x + 0.2, s_p / s_p.sum(), width=0.4, label="purity head")
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_channels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("attribution mass fraction")
    ax.set_title("Chemical-sensor attribution by head")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "shap_sensor_importance.png"), dpi=120)
    plt.close(fig)

    # ---------------- console summary --------------------------
    print("\n" + "=" * 60)
    print("SHAP SUMMARY  (mean |SHAP| attribution mass)")
    print("=" * 60)
    for head, s in (("MATERIAL", mat_summary), ("PURITY", pur_summary)):
        g = s["group_mass_fraction"]
        print(f"\n[{head}]  explainer={s['explainer']}")
        print(f"  spectrum(SNV)      {g['spectrum_snv']:.3f}")
        print(f"  spectrum(deriv)    {g['spectrum_derivative']:.3f}")
        print(f"  chemical sensors   {g['chemical_sensors']:.3f}")
        print("  top sensors        " + ", ".join(
            list(s["sensor_importance"])[:4]))
        print("  top wavelengths    " + ", ".join(
            f"{w['wavelength_nm']:.0f}nm" for w in s["top_wavelengths"][:5]))

    print("\nSaved JSON + PNG to", OUT_DIR)


def _top_named(vec, names, k):
    order = np.argsort(np.abs(vec))[::-1][:k]
    return [{"feature": names[i], "shap_value": float(vec[i])} for i in order]


if __name__ == "__main__":
    main()
