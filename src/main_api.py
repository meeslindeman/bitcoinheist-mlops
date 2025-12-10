import json
import pandas as pd

from time import time
from flask import Flask, request, jsonify, render_template
from functools import cache
from prometheus_flask_exporter import PrometheusMetrics

from configs.configs import PathsConfig, RunConfig, ModelConfig
from src.features import get_features
from src.model import Model
from src.spark_utils import get_spark_session
from src.telemetry.training import init_training_metrics_from_file
from src.mlflow_utils import get_latest_run_id
from src.telemetry.api import (
    PREDICTION_REQUESTS,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_LABELS
)

app = Flask(__name__, template_folder="templates", static_folder="static")
metrics = PrometheusMetrics(app, defaults_prefix="bitcoin_heist")
metrics.info(
    "bitcoin_heist_model_version",
    "Model version information",
    experiment_name=RunConfig.experiment_name,
    run_name=RunConfig.run_name,
    model_name=ModelConfig.model_name,
)

spark = get_spark_session(app_name="bitcoin-heist-api")

try:
    with open(PathsConfig.feature_columns_path, "r") as f:
        FEATURE_COLUMNS = json.load(f)
except FileNotFoundError:
    app.logger.warning("Feature columns file not found at startup, run the training pipeline before requesting predictions.")
    FEATURE_COLUMNS = None

try:
    init_training_metrics_from_file()
except Exception as e:
    app.logger.warning(f"Could not initialize training metrics at startup: {e}")

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@cache
def get_model() -> Model:
    run_id = get_latest_run_id(RunConfig.run_name)
    if run_id is None:
        raise RuntimeError("No trained model run found in MLflow.")

    model = Model()
    model.load_model_from_mlflow(run_id)
    return model

@app.post("/reload")
def reload_model():
    try:
        get_model.cache_clear()
        _ = get_model()
        try:
            init_training_metrics_from_file()
        except Exception as e:
            app.logger.warning(f"Failed to re-init training metrics: {e}")

        return jsonify({"status": "ok", "message": "Model reloaded from MLflow"}), 200

    except Exception as e:
        app.logger.exception("Failed to reload model from MLflow")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.post("/predict")
def predict():
    start_time = time()
    PREDICTION_REQUESTS.inc()

    try:
        # note: get model lazy + cached
        try:
            model = get_model()
        except RuntimeError as e:
            app.logger.warning(f"Prediction request but no model available: {e}")
            PREDICTION_ERRORS.inc()
            return jsonify({"error": "No trained model available yet."}), 503
        
        if FEATURE_COLUMNS is None:
            app.logger.warning("Prediction request but feature columns are not available.")
            PREDICTION_ERRORS.inc()
            return jsonify({"error": "Model features not initialized yet."}), 503

        # note: validate payload
        data = request.get_json()
        if data is None:
            PREDICTION_ERRORS.inc()
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        input_pd = pd.DataFrame([data])

        if "address" in input_pd.columns:
            input_pd = input_pd.drop(columns=["address"])

        # note: extract features 
        input_spark = spark.createDataFrame(input_pd)
        features_spark = get_features(input_spark)
        features_pd = features_spark.toPandas()

        if "is_ransomware" in features_pd.columns:
            features_pd = features_pd.drop(columns=["is_ransomware"])

        for col in FEATURE_COLUMNS:
            if col not in features_pd.columns:
                features_pd[col] = 0.0

        features_pd = features_pd[FEATURE_COLUMNS]

        # note: make prediction
        proba = float(model.predict_proba(features_pd)[0, 1])
        prediction = "ransomware" if proba >= 0.5 else "clean"

        PREDICTION_LABELS.labels(prediction=prediction).inc()

        return (
            jsonify({"prediction": prediction, "probability": proba}),
            200,
        )

    except Exception as e:
        app.logger.exception("Unhandled error during prediction")
        PREDICTION_ERRORS.inc()
        # Let Flask return 500 with message (or hide message if you prefer)
        return jsonify({"error": "Internal server error"}), 500

    finally:
        runtime = time() - start_time
        PREDICTION_LATENCY.observe(runtime)

# note: simple UI 
@app.get("/")
def index():
    return render_template("index.html")