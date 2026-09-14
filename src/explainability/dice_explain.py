import pandas as pd
import joblib
import dice_ml
from dice_ml import Dice

DATA_FILE = "data/processed/material_industry_compatibility.csv"
MODEL_FILE = "models/compatibility/random_forest.joblib"
PREPROCESSOR_FILE = "models/compatibility/preprocessor.joblib"
OUTPUT_FILE = "data/processed/dice_counterfactuals.csv"

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

CATEGORICAL = [
    "chemical_name",
    "primary_naics"
]

CONTINUOUS = [
    "records",
    "total_release",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed",
    "treatment_rate",
    "management_rate"
]

FEATURES_TO_VARY = [
    "records",
    "total_release",
    "total_recycled",
    "total_treated",
    "total_energy_recovery",
    "total_managed",
    "treatment_rate",
    "management_rate"
]

CLASS_MAP = {
    "HIGH": 0,
    "LOW": 1,
    "MEDIUM": 2
}

REVERSE_MAP = {
    0: "HIGH",
    1: "LOW",
    2: "MEDIUM"
}

df = pd.read_csv(DATA_FILE)

df["chemical_name"] = df["chemical_name"].astype(str)
df["primary_naics"] = df["primary_naics"].astype(str)

df["compatibility_class"] = df["compatibility_class"].map(CLASS_MAP)

df = df[FEATURES + ["compatibility_class"]].dropna()

model = joblib.load(MODEL_FILE)
preprocessor = joblib.load(PREPROCESSOR_FILE)


class CompatibilityModel:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURES)
        X["chemical_name"] = X["chemical_name"].astype(str)
        X["primary_naics"] = X["primary_naics"].astype(str)

        processed = self.preprocessor.transform(X)

        return pd.DataFrame(
            processed,
            columns=self.preprocessor.get_feature_names_out()
        )

    def predict(self, X):
        processed = self.transform(X)
        predictions = self.model.predict(processed)
        return [CLASS_MAP[p] for p in predictions]

    def predict_proba(self, X):
        processed = self.transform(X)
        return self.model.predict_proba(processed)


wrapped_model = CompatibilityModel(model, preprocessor)

data = dice_ml.Data(
    dataframe=df,
    continuous_features=CONTINUOUS,
    categorical_features=CATEGORICAL,
    outcome_name="compatibility_class"
)

dice_model = dice_ml.Model(
    model=wrapped_model,
    backend="sklearn",
    model_type="classifier"
)

explainer = Dice(data, dice_model, method="random")

# Find a LOW or MEDIUM example
candidate_indices = []

for i in range(len(df)):
    row = df[FEATURES].iloc[[i]]
    prediction = wrapped_model.predict(row)[0]

    if prediction in [CLASS_MAP["LOW"], CLASS_MAP["MEDIUM"]]:
        candidate_indices.append(i)

    if len(candidate_indices) >= 1:
        break

index = candidate_indices[0]
query = df[FEATURES].iloc[[index]]

current_prediction = wrapped_model.predict(query)[0]

print("Original prediction:", REVERSE_MAP[current_prediction])
print("Target prediction: HIGH")

permitted_range = {}

for feature in CONTINUOUS:
    permitted_range[feature] = [
        float(df[feature].min()),
        float(df[feature].max())
    ]

counterfactuals = explainer.generate_counterfactuals(
    query,
    total_CFs=3,
    desired_class=CLASS_MAP["HIGH"],
    features_to_vary=FEATURES_TO_VARY,
    permitted_range=permitted_range
)

result = counterfactuals.cf_examples_list[0].final_cfs_df.copy()

result["compatibility_class"] = result["compatibility_class"].map(
    REVERSE_MAP
)

print("\nOriginal instance:")
print(query)

print("\nCounterfactuals:")
print(result)

result.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)