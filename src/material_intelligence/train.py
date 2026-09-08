"""
Train the multi-task 1-D CNN (material classification + purity regression) and
benchmark it against a RandomForest baseline on the same features.

Inputs
------
data/processed/mi_dataset.npz            (from material_intelligence/preprocess.py)
models/material_intelligence/preprocess.joblib

Outputs
-------
models/material_intelligence/cnn.pt
results/material_intelligence/cnn_results.json
results/material_intelligence/training_curve.png
results/material_intelligence/confusion_matrix.png
results/material_intelligence/purity_scatter.png
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
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from sklearn.ensemble import (  # noqa: E402
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from cnn_model import MultiTaskCNN1D  # noqa: E402

DATA_NPZ = "data/processed/mi_dataset.npz"
TRANSFORM = "models/material_intelligence/preprocess.joblib"
MODEL_OUT = "models/material_intelligence/cnn.pt"
RESULTS_OUT = "results/material_intelligence/cnn_results.json"
CURVE_PNG = "results/material_intelligence/training_curve.png"
CM_PNG = "results/material_intelligence/confusion_matrix.png"
SCATTER_PNG = "results/material_intelligence/purity_scatter.png"

SEED = 42
EPOCHS = 80
BATCH = 128
LR = 1e-3
PURITY_LOSS_WEIGHT = 6.0
PATIENCE = 12

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_split(d, name):
    return (
        d[f"Xspec_{name}"], d[f"Xsens_{name}"],
        d[f"ymat_{name}"], d[f"ypur_{name}"],
    )


def make_loader(Xspec, Xsens, ymat, ypur, shuffle):
    ds = TensorDataset(
        torch.tensor(Xspec, dtype=torch.float32),
        torch.tensor(Xsens, dtype=torch.float32),
        torch.tensor(ymat, dtype=torch.long),
        torch.tensor(ypur, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    logits_all, pur_all, ymat_all, ypur_all = [], [], [], []
    with torch.no_grad():
        for xs, xc, ym, yp in loader:
            xs, xc = xs.to(device), xc.to(device)
            lg, pu = model(xs, xc)
            logits_all.append(lg.cpu())
            pur_all.append(pu.cpu())
            ymat_all.append(ym)
            ypur_all.append(yp)
    logits = torch.cat(logits_all).numpy()
    pur = torch.cat(pur_all).numpy()
    ymat = torch.cat(ymat_all).numpy()
    ypur = torch.cat(ypur_all).numpy()
    pred = logits.argmax(1)
    return {
        "y_true_mat": ymat, "y_pred_mat": pred,
        "y_true_pur": ypur, "y_pred_pur": pur,
        "acc": accuracy_score(ymat, pred),
        "macro_f1": f1_score(ymat, pred, average="macro"),
        "pur_mae": mean_absolute_error(ypur, pur),
    }


def main():
    if not os.path.exists(DATA_NPZ):
        raise FileNotFoundError(
            f"{DATA_NPZ} not found - run material_intelligence/preprocess.py first."
        )

    meta = joblib.load(TRANSFORM)
    material_names = meta["material_names"]
    sensor_channels = meta["sensor_channels"]

    d = np.load(DATA_NPZ)
    Xspec_tr, Xsens_tr, ymat_tr, ypur_tr = load_split(d, "train")
    Xspec_va, Xsens_va, ymat_va, ypur_va = load_split(d, "val")
    Xspec_te, Xsens_te, ymat_te, ypur_te = load_split(d, "test")

    spec_len = Xspec_tr.shape[2]
    n_sensors = Xsens_tr.shape[1]
    n_classes = len(material_names)

    device = torch.device("cpu")
    print(f"device={device} | train={len(Xspec_tr)} val={len(Xspec_va)} "
          f"test={len(Xspec_te)} | spec_len={spec_len} classes={n_classes}")

    train_loader = make_loader(Xspec_tr, Xsens_tr, ymat_tr, ypur_tr, True)
    val_loader = make_loader(Xspec_va, Xsens_va, ymat_va, ypur_va, False)
    test_loader = make_loader(Xspec_te, Xsens_te, ymat_te, ypur_te, False)

    model = MultiTaskCNN1D(spec_len, n_sensors, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNN trainable parameters: {n_params:,}")

    # class weights for imbalance
    counts = np.bincount(ymat_tr, minlength=n_classes).astype(np.float64)
    cw = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)),
                      dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=cw)
    huber = nn.SmoothL1Loss()

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_pur_mae": []}
    best_val = np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run = 0.0
        for xs, xc, ym, yp in train_loader:
            xs, xc, ym, yp = xs.to(device), xc.to(device), ym.to(device), yp.to(device)
            opt.zero_grad()
            lg, pu = model(xs, xc)
            loss = ce(lg, ym) + PURITY_LOSS_WEIGHT * huber(pu, yp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * len(ym)
        train_loss = run / len(ymat_tr)

        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for xs, xc, ym, yp in val_loader:
                xs, xc, ym, yp = xs.to(device), xc.to(device), ym.to(device), yp.to(device)
                lg, pu = model(xs, xc)
                vrun += (ce(lg, ym) + PURITY_LOSS_WEIGHT * huber(pu, yp)).item() * len(ym)
        val_loss = vrun / len(ymat_va)
        vm = evaluate(model, val_loader, device)
        sched.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(vm["acc"])
        history["val_pur_mae"].append(vm["pur_mae"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| val_acc {vm['acc']:.4f} | val_pur_MAE {vm['pur_mae']:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # ---------------- test evaluation --------------------------------
    tm = evaluate(model, test_loader, device)
    cnn_acc = accuracy_score(tm["y_true_mat"], tm["y_pred_mat"])
    cnn_f1 = f1_score(tm["y_true_mat"], tm["y_pred_mat"], average="macro")
    cnn_report = classification_report(
        tm["y_true_mat"], tm["y_pred_mat"],
        target_names=material_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(tm["y_true_mat"], tm["y_pred_mat"])
    pur_mae = mean_absolute_error(tm["y_true_pur"], tm["y_pred_pur"])
    pur_rmse = float(np.sqrt(mean_squared_error(tm["y_true_pur"], tm["y_pred_pur"])))
    pur_r2 = r2_score(tm["y_true_pur"], tm["y_pred_pur"])

    print("\n" + "=" * 60)
    print("CNN TEST RESULTS")
    print("=" * 60)
    print(f"material  accuracy : {cnn_acc:.4f}")
    print(f"material  macro-F1 : {cnn_f1:.4f}")
    print(f"purity    MAE      : {pur_mae:.4f}")
    print(f"purity    RMSE     : {pur_rmse:.4f}")
    print(f"purity    R2       : {pur_r2:.4f}")

    # ---------------- RandomForest baseline -------------------------
    print("\nTraining RandomForest baseline on flattened features...")
    Xtr_flat = np.concatenate(
        [Xspec_tr.reshape(len(Xspec_tr), -1), Xsens_tr], axis=1)
    Xte_flat = np.concatenate(
        [Xspec_te.reshape(len(Xspec_te), -1), Xsens_te], axis=1)

    rf_c = RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced")
    rf_c.fit(Xtr_flat, ymat_tr)
    rf_pred = rf_c.predict(Xte_flat)
    rf_acc = accuracy_score(ymat_te, rf_pred)
    rf_f1 = f1_score(ymat_te, rf_pred, average="macro")

    rf_r = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf_r.fit(Xtr_flat, ypur_tr)
    rf_pur = rf_r.predict(Xte_flat)
    rf_pur_mae = mean_absolute_error(ypur_te, rf_pur)
    rf_pur_r2 = r2_score(ypur_te, rf_pur)

    print(f"RF material accuracy : {rf_acc:.4f}  (macro-F1 {rf_f1:.4f})")
    print(f"RF purity   MAE      : {rf_pur_mae:.4f}  (R2 {rf_pur_r2:.4f})")

    # ---------------- plots ----------------------------------------
    os.makedirs("results/material_intelligence", exist_ok=True)

    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, history["train_loss"], label="train loss")
    ax1.plot(ep, history["val_loss"], label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(ep, history["val_acc"], "g--", label="val acc")
    ax2.set_ylabel("val accuracy")
    fig.tight_layout(); fig.savefig(CURVE_PNG, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(material_names, rotation=90, fontsize=7)
    ax.set_yticklabels(material_names, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"CNN confusion matrix (acc={cnn_acc:.3f})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(CM_PNG, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(tm["y_true_pur"], tm["y_pred_pur"], s=6, alpha=0.3)
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("true purity"); ax.set_ylabel("predicted purity")
    ax.set_title(f"CNN purity (MAE={pur_mae:.3f}, R2={pur_r2:.3f})")
    fig.tight_layout(); fig.savefig(SCATTER_PNG, dpi=120); plt.close(fig)

    # ---------------- save model + results ------------------------
    os.makedirs("models/material_intelligence", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "spec_len": int(spec_len),
                "n_sensors": int(n_sensors),
                "n_classes": int(n_classes),
            },
            "material_names": material_names,
            "sensor_channels": sensor_channels,
        },
        MODEL_OUT,
    )

    results = {
        "cnn": {
            "trainable_parameters": int(n_params),
            "epochs_run": len(history["train_loss"]),
            "material_accuracy": float(cnn_acc),
            "material_macro_f1": float(cnn_f1),
            "purity_mae": float(pur_mae),
            "purity_rmse": float(pur_rmse),
            "purity_r2": float(pur_r2),
            "classification_report": cnn_report,
            "confusion_matrix": cm.tolist(),
        },
        "random_forest_baseline": {
            "material_accuracy": float(rf_acc),
            "material_macro_f1": float(rf_f1),
            "purity_mae": float(rf_pur_mae),
            "purity_r2": float(rf_pur_r2),
        },
        "data": {
            "train": int(len(Xspec_tr)),
            "val": int(len(Xspec_va)),
            "test": int(len(Xspec_te)),
            "materials": material_names,
            "sensor_channels": sensor_channels,
            "spectral_channels": meta["spectral_channels"],
        },
        "notes": (
            "Synthetic NIR + chemical-sensor data from the digital-twin "
            "simulator. Multi-task 1-D CNN shares a convolutional spectral "
            "backbone between a material-classification head and a purity-"
            "regression head. Purity is a continuous contamination-blend label."
        ),
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    for p in (MODEL_OUT, RESULTS_OUT, CURVE_PNG, CM_PNG, SCATTER_PNG):
        print(" ", p)


if __name__ == "__main__":
    main()
