import pandas as pd
import sys
import os
import mlflow
import joblib

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_model(model_name: str, stage: str = "None", use_mlflow: bool = True):
    if use_mlflow:
        if stage != "None":
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = model_name
        return mlflow.sklearn.load_model(model_uri)
    else:
        return joblib.load(model_name)

def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict_proba(df)[:, 1]
    return pd.DataFrame({"risk_probability": preds})

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scripts/predict.py <data_path> <model_path_or_name> <stage_or_None>")
        sys.exit(1)

    data_path = sys.argv[1]
    model_path_or_name = sys.argv[2]
    stage = sys.argv[3]

    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        sys.exit(1)

    df = load_data(data_path)

    # Determine whether to use MLflow or joblib (auto-detect by file extension)
    use_mlflow = not model_path_or_name.endswith(".pkl")
    model = load_model(model_path_or_name, stage, use_mlflow)

    predictions = predict(model, df)

    print("\n✅ Sample Predictions:\n", predictions.head())
    predictions.to_csv("predictions.csv", index=False)
    print("📁 Predictions saved to predictions.csv")
