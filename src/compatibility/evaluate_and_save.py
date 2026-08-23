"""
Evaluates the compatibility model and outputs true feature importances.
"""
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os

def evaluate():
    print("Loading model and honest test data...")
    X_test = pd.read_csv("data/processed/X_test_compat.csv")
    y_test = pd.read_csv("data/processed/y_test_compat.csv").squeeze()
    
    model = joblib.load("models/compatibility/random_forest.joblib")
    
    preds = model.predict(X_test)
    
    print("\n" + "=" * 60)
    print("COMPATIBILITY RISK PREDICTION (HONEST EVALUATION)")
    print("=" * 60)
    
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    print("\nTop 15 feature importances:")
    # Now we have real column names!
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    print(importances.sort_values(ascending=False).head(15))
    
    os.makedirs("results/compatibility", exist_ok=True)
    results = {"accuracy": acc}
    with open("results/compatibility/results.json", "w") as f:
        json.dump(results, f)
    print("\nSaved: results/compatibility/results.json")

if __name__ == "__main__":
    evaluate()