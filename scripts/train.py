import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature

# Load processed data
print("-_-_ Loading data...")
data_path = os.path.join("data", "processed", "processed_data.csv")
df = pd.read_csv(data_path)

# Define features and target
X = df[['Total_Amount', 'Avg_Amount', 'Std_Amount', 'Num_Transactions', 'Transaction_Hour',
        'ProductCategory', 'ChannelId', 'PricingStrategy']]
y = df['is_high_risk']

# Dummy encoding for categorical features
X = pd.get_dummies(X)

# Save feature names
os.makedirs("models", exist_ok=True)
with open("models/feature_names.txt", "w") as f:
    for name in X.columns:
        f.write(f"{name}\n")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setup imputation and scaling in pipeline
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Start MLflow experiment
mlflow.set_experiment("credit-risk-model")

with mlflow.start_run():
    # Logistic Regression
    print("// Training LogisticRegression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    auc_lr = roc_auc_score(y_test, y_pred_lr)
    print(f"//LogisticRegression ROC-AUC: {auc_lr:.4f}")

    mlflow.log_metric("logreg_roc_auc", auc_lr)

    signature_lr = infer_signature(X_test, y_pred_lr)
    mlflow.sklearn.log_model(
        lr,
        name="logistic-regression-model",
        input_example=pd.DataFrame(X_test[:5], columns=X.columns),
        signature=signature_lr
    )

    # XGBoost
    print("//Training XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    auc_xgb = roc_auc_score(y_test, y_pred_xgb)
    print(f"// XGBoost ROC-AUC: {auc_xgb:.4f}")

    mlflow.log_metric("xgboost_roc_auc", auc_xgb)

    signature_xgb = infer_signature(X_test, y_pred_xgb)
    mlflow.xgboost.log_model(
        xgb,
        name="xgboost-model",
        input_example=pd.DataFrame(X_test[:5], columns=X.columns),
        signature=signature_xgb
    )

print(" Done Abelo Training and logging completed.")
