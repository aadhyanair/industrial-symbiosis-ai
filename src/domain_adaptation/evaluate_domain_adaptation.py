import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import silhouette_score


REAL = "data/processed/material_industry_compatibility.csv"
SYN = "data/processed/synthetic_waste_profiles_v5.csv"

PREPROCESSOR = "models/compatibility/preprocessor.joblib"
MODEL = "models/domain_adaptation/domain_encoder.pt"


FEATURES = [
    "chemical_name",
    "primary_naics",
    "records",
    "total_release",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed",
    "treatment_rate",
    "management_rate"
]


class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        classes
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        self.classifier = nn.Linear(
            embedding_dim,
            classes
        )

    def forward(self, x):

        z = F.normalize(
            self.encoder(x),
            dim=1
        )

        y = self.classifier(z)

        return z, y


def clean_data(df):

    df = df.copy()

    df["chemical_name"] = (
        df["chemical_name"]
        .fillna("UNKNOWN")
    )

    df["primary_naics"] = (
        df["primary_naics"]
        .fillna("UNKNOWN")
    )

    for col in FEATURES[2:]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    return df[FEATURES]


def main():

    print("Loading data...")

    real = pd.read_csv(REAL)
    synthetic = pd.read_csv(SYN)

    real = real.dropna(
        subset=["compatibility_class"]
    )

    X_real_raw = clean_data(real)
    X_syn_raw = clean_data(synthetic)

    preprocessor = joblib.load(
        PREPROCESSOR
    )

    X_real = preprocessor.transform(
        X_real_raw
    )

    X_syn = preprocessor.transform(
        X_syn_raw
    )

    X_real = np.asarray(
        X_real,
        dtype=np.float32
    )

    X_syn = np.asarray(
        X_syn,
        dtype=np.float32
    )

    classes = sorted(
        real["compatibility_class"]
        .astype(str)
        .unique()
    )

    class_to_id = {
        c: i
        for i, c in enumerate(classes)
    }

    y_real = np.array([
        class_to_id[x]
        for x in
        real["compatibility_class"]
        .astype(str)
    ])

    _, test_indices = train_test_split(
        np.arange(len(X_real)),
        test_size=0.2,
        random_state=42,
        stratify=y_real
    )

    X_test = X_real[test_indices]
    y_test = y_real[test_indices]

    checkpoint = torch.load(
        MODEL,
        map_location="cpu",
        weights_only=False
    )

    model = Encoder(
        checkpoint["input_dim"],
        checkpoint["embedding_dim"],
        len(checkpoint["classes"])
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():

        test_z, test_pred = model(
            torch.tensor(X_test)
        )

        syn_z, _ = model(
            torch.tensor(X_syn)
        )

    predictions = torch.argmax(
        test_pred,
        dim=1
    ).numpy()

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 55)
    print("DOMAIN ADAPTATION TEST RESULTS")
    print("=" * 55)

    print(
        f"\nHeld-out real accuracy: "
        f"{accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=classes
        )
    )

    test_embeddings = test_z.numpy()
    synthetic_embeddings = syn_z.numpy()

    mean_distance = np.linalg.norm(
        test_embeddings.mean(axis=0)
        - synthetic_embeddings.mean(axis=0)
    )

    print(
        f"Test/Synthetic embedding "
        f"mean distance: {mean_distance:.4f}"
    )

    combined = np.vstack([
        test_embeddings,
        synthetic_embeddings
    ])

    domain_labels = np.array(
        [0] * len(test_embeddings)
        + [1] * len(synthetic_embeddings)
    )

    silhouette = silhouette_score(
        combined,
        domain_labels
    )

    print(
        f"Test/Synthetic domain "
        f"silhouette: {silhouette:.4f}"
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()