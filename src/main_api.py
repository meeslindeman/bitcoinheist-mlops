import json
import pandas as pd

from time import time
from flask import Flask, request, jsonify

from configs.configs import PathsConfig
from src.features import get_features
from src.model import Model
from src.spark_utils import get_spark_session
from src.telemetry.training import init_training_metrics_from_file
from src.telemetry.api import (
    PREDICTION_REQUESTS,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_LABELS,
    metrics_response,
)

app = Flask(__name__)
spark = get_spark_session(app_name="bitcoin-heist-api")

with open(PathsConfig.feature_columns_path, "r") as f:
    FEATURE_COLUMNS = json.load(f)

model = Model()
model.load_model_local()

init_training_metrics_from_file()

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.get("/metrics")
def metrics():
    return metrics_response()


@app.post("/predict")
def predict():
    start_time = time()
    PREDICTION_REQUESTS.inc()

    try:
        payload = request.get_json()
        if payload is None:
            PREDICTION_ERRORS.inc()
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        input_pd = pd.DataFrame([payload])

        # note: address is not used for features
        if "address" in input_pd.columns:
            input_pd = input_pd.drop(columns=["address"])

        # note: apply the same feature logic as in training
        input_spark = spark.createDataFrame(input_pd)
        features_spark = get_features(input_spark)
        features_pd = features_spark.toPandas()

        # note: ensure we don't have labels here
        if "is_ransomware" in features_pd.columns:
            features_pd = features_pd.drop(columns=["is_ransomware"])

        # note: align to training feature columns: add missing columns with 0.0 values
        for col in FEATURE_COLUMNS:
            if col not in features_pd.columns:
                features_pd[col] = 0.0

        features_pd = features_pd[FEATURE_COLUMNS]

        proba = float(model.predict_proba(features_pd)[0, 1])
        prediction = "ransomware" if proba >= 0.5 else "clean"

        PREDICTION_LABELS.labels(prediction=prediction).inc()

        return jsonify({"ransomware_probability": proba, "prediction": prediction}), 200
    
    except Exception:
        PREDICTION_ERRORS.inc()
        raise

    finally:
        runtime = time() - start_time
        PREDICTION_LATENCY.observe(runtime)