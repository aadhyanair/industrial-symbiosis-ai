import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

REAL = "data/processed/material_industry_compatibility.csv"
SYN = "data/processed/synthetic_waste_profiles_v5.csv"
PREPROCESSOR = "models/compatibility/preprocessor.joblib"
RF_MODEL = "models/compatibility/random_forest.joblib"
OUTPUT = "models/domain_adaptation/domain_encoder.pt"

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
    def __init__(self, input_dim, embedding_dim=64, classes=3):
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


def contrastive_loss(
    real_z,
    syn_z,
    real_y,
    syn_y,
    temperature=0.2
):
    similarity = torch.mm(
        real_z,
        syn_z.T
    ) / temperature

    positive = (
        real_y.unsqueeze(1)
        == syn_y.unsqueeze(0)
    ).float()

    log_prob = F.log_softmax(
        similarity,
        dim=1
    )

    positive_count = positive.sum(
        dim=1
    ).clamp(min=1)

    loss = -(
        positive * log_prob
    ).sum(dim=1) / positive_count

    return loss.mean()


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

    X_real = clean_data(real)
    X_syn = clean_data(synthetic)

    print("Loading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR
    )

    rf = joblib.load(
        RF_MODEL
    )

    print("Transforming data...")

    X_real = preprocessor.transform(
        X_real
    )

    X_syn = preprocessor.transform(
        X_syn
    )

    X_real = np.asarray(
        X_real,
        dtype=np.float32
    )

    X_syn = np.asarray(
        X_syn,
        dtype=np.float32
    )

    print("Real shape:", X_real.shape)
    print("Synthetic shape:", X_syn.shape)

    classes = sorted(
        real["compatibility_class"]
        .astype(str)
        .unique()
    )

    class_to_id = {
        c: i
        for i, c in enumerate(classes)
    }

    real_labels = (
        real["compatibility_class"]
        .astype(str)
        .values
    )

    y_real = np.array([
        class_to_id[x]
        for x in real_labels
    ])

    print("\nCompatibility classes:")

    for c, i in class_to_id.items():
        print(i, "=", c)

    print("\nGenerating synthetic pseudo-labels...")

    syn_pred = rf.predict(
        pd.DataFrame(
            X_syn,
            columns=preprocessor.get_feature_names_out()
        )
    )

    syn_pred = np.asarray(
        syn_pred
    ).astype(str)

    y_syn = np.array([
        class_to_id.get(x, 0)
        for x in syn_pred
    ])

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_real,
            y_real,
            test_size=0.2,
            random_state=42,
            stratify=y_real
        )
    )

    n = min(
        len(X_train),
        len(X_syn)
    )

    X_train = X_train[:n]
    y_train = y_train[:n]

    X_syn = X_syn[:n]
    y_syn = y_syn[:n]

    device = torch.device("cpu")

    model = Encoder(
        input_dim=X_real.shape[1],
        embedding_dim=64,
        classes=len(classes)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.long
        ),
        torch.tensor(
            X_syn,
            dtype=torch.float32
        ),
        torch.tensor(
            y_syn,
            dtype=torch.long
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )

    print("\nTraining corrected domain adaptation model...")

    epochs = 30

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for xr, yr, xs, ys in loader:

            zr, pr = model(xr)
            zs, ps = model(xs)

            classification_loss = (
                F.cross_entropy(
                    pr,
                    yr
                )
            )

            contrastive = contrastive_loss(
                zr,
                zs,
                yr,
                ys
            )

            domain_loss = torch.mean(
                (
                    zr.mean(0)
                    - zs.mean(0)
                ) ** 2
            )

            loss = (
                classification_loss
                + 0.5 * contrastive
                + 0.1 * domain_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Loss: "
                f"{total_loss / len(loader):.4f}"
            )

    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "input_dim":
                X_real.shape[1],
            "embedding_dim":
                64,
            "classes":
                classes
        },
        OUTPUT
    )

    print("\nModel saved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()