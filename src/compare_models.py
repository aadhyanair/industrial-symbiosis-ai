import pandas as pd


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Tuned XGBoost"
    ],
    "Accuracy": [
        0.6466,
        0.6288,
        0.6739,
        0.6907
    ],
    "Balanced Accuracy": [
        0.6841,
        0.6782,
        0.7131,
        0.7264
    ]
})

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Balanced Accuracy": "{:.4f}".format
        }
    )
)

print("\nBest model based on balanced accuracy:")

best = results.loc[
    results["Balanced Accuracy"].idxmax()
]

print(
    f"{best['Model']} "
    f"({best['Balanced Accuracy']:.4f})"
)