# -*- coding: utf-8 -*-
"""
House Price Prediction with:
- Sklearn Pipeline
- RandomForestRegressor
- MLflow Tracking
- Model Promotion Logic Gate
- Explicit Production Model Export
"""

import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# Configuration
# =========================
EXPERIMENT_NAME = "House_Price_RF_Pipeline"
MODEL_NAME = "HousePriceRandomForest"
BASELINE_R2 = 0.75
EXPORT_PATH = Path("models/production_model.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# =========================
# MLflow Setup
# =========================
mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# Load Dataset
# =========================
data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# =========================
# Pipeline Definition
# =========================
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# =========================
# Training + Tracking
# =========================
with mlflow.start_run() as run:

    pipeline.fit(X_train, y_train)

    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("baseline_r2", BASELINE_R2)

    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("test_mae", test_mae)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Baseline R2: {BASELINE_R2}")

# =========================
# Logic Gate + Export
# =========================
if test_r2 >= BASELINE_R2:
    print("✅ Model passed baseline. Promoting to Production...")

    client = mlflow.tracking.MlflowClient()

    latest_version = client.get_latest_versions(
        MODEL_NAME, stages=["None"]
    )[0].version

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🚀 Model version {latest_version} promoted to Production")

    # -------------------------
    # Explicit Export for Inference
    # -------------------------
    print("📦 Exporting Production model for inference...")

    prod_model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/Production"
    )

    EXPORT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(prod_model, EXPORT_PATH)

    print(f"✅ Production model exported to {EXPORT_PATH.resolve()}")

else:
    print("❌ Model did NOT meet baseline. Promotion skipped.")
