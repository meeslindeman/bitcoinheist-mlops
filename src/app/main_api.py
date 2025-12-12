import json
import pandas as pd

from time import time
from flask import Flask, request, jsonify, render_template
from functools import cache
from prometheus_flask_exporter import PrometheusMetrics

from configs.configs import PathsConfig, RunConfig, ModelConfig
from src.pipeline.features import get_features
from src.pipeline.feature_extractor import FeatureExtractor
from src.model.model import Model
from src.utils.spark_utils import get_spark_session
from src.utils.mlflow_utils import get_latest_run_id
from src.telemetry.api import PREDICTION_REQUESTS, PREDICTION_ERRORS, PREDICTION_LATENCY, PREDICTION_LABELS


app = Flask(__name__, template_folder="templates", static_folder="static")
metrics = PrometheusMetrics(app, defaults_prefix="bitcoin_heist")
metrics.info(
    "bitcoin_heist_model_version",
    "Model version information",
    experiment_name=RunConfig.experiment_name,
    run_name=RunConfig.run_name,
    model_name=ModelConfig.model_name,
)

# note: lazy import for testing purposes
spark = None

def get_spark():
    global spark
    if spark is None:
        spark = get_spark_session(app_name="bitcoin-heist-api")
    return spark

try:
    with open(PathsConfig.feature_columns_path, "r") as f:
        feature_columns = json.load(f)
except FileNotFoundError:
    app.logger.warning("Feature columns file not found at startup, run the training pipeline before requesting predictions.")
    feature_columns = None

try:
    feature_extractor = FeatureExtractor.load_state(PathsConfig.feature_extractor_state_dir)
except FileNotFoundError:
    app.logger.warning("Feature extractor state not found at startup, run the feature engineering step before requesting predictions.")
    feature_extractor = None

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
        
        # note: feature columns available
        if feature_columns is None:
            app.logger.warning("Prediction request but feature columns are not available.")
            PREDICTION_ERRORS.inc()
            return jsonify({"error": "Model features not initialized yet."}), 503
        
        # note: feature extractor available
        if feature_extractor is None:
            app.logger.warning("Prediction request but z-score feature extractor is not available.")
            PREDICTION_ERRORS.inc()
            return jsonify({"error": "Feature extractor not initialized yet."}), 503

        # note: validate prediction input
        # note: silent=True so we return None for invalid JSON and handle it ourselves
        data = request.get_json(silent=True) 
        if data is None:
            PREDICTION_ERRORS.inc()
            return jsonify({"error": "Invalid or missing JSON body"}), 400
        
        # note: validate required fields
        required_fields = ["year", "day", "length", "weight", "count", "looped", "neighbors", "income"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            PREDICTION_ERRORS.inc()
            return (jsonify({"error": "Missing required input fields", "missing_fields": missing}), 400)

        input_pd = pd.DataFrame([data])

        if "address" in input_pd.columns:
            input_pd = input_pd.drop(columns=["address"])

        # note: stateless Spark feature engineering: log/ratio/temporal
        input_spark = get_spark().createDataFrame(input_pd)
        features_spark = get_features(input_spark)

        # note: stateful feature extraction: z-score normalization
        features_spark = feature_extractor.get_features(features_spark)

        # note: drop raw columns that the model was not trained on
        drop_cols = ["year", "income", "length"]
        existing_drop_cols = [c for c in drop_cols if c in features_spark.columns]
        if existing_drop_cols:
            features_spark = features_spark.drop(*existing_drop_cols)

        features_pd = features_spark.toPandas()

        # note: drop target column if present to be safe
        if "is_ransomware" in features_pd.columns:
            features_pd = features_pd.drop(columns=["is_ransomware"])

        # note: align to training feature columns: add missing and reorder
        for col in feature_columns:
            if col not in features_pd.columns:
                features_pd[col] = 0.0
        features_pd = features_pd[feature_columns]

        # note: make prediction
        proba = float(model.predict_proba(features_pd)[0, 1])
        prediction = "ransomware" if proba >= 0.5 else "clean"

        PREDICTION_LABELS.labels(prediction=prediction).inc()

        return (jsonify({"prediction": prediction, "probability": proba}), 200)

    except Exception as e:
        app.logger.exception("Unhandled error during prediction")
        PREDICTION_ERRORS.inc()
        return jsonify({"error": "Internal server error"}), 500

    finally:
        runtime = time() - start_time
        PREDICTION_LATENCY.observe(runtime)

# note: simple UI 
@app.get("/")
def index():
    return render_template("index.html")