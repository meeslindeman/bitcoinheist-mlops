import json
import pandas as pd
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession

from configs.configs import PathsConfig
from src.features import get_features
from src.model import Model


def get_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("BitcoinHeistAPI")
        .master("local[2]")
        .getOrCreate()
    )
    return spark


app = Flask(__name__)
spark = get_spark_session()

# load feature columns metadata
with open(PathsConfig.feature_columns_path, "r") as f:
    FEATURE_COLUMNS = json.load(f)

# load trained model from local filesystem
model = Model()
model.load_model_local()


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    payload = request.get_json()
    if payload is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    input_pd = pd.DataFrame([payload])

    # address is not used for features
    if "address" in input_pd.columns:
        input_pd = input_pd.drop(columns=["address"])

    # use Spark to apply the same feature logic as in training
    input_spark = spark.createDataFrame(input_pd)
    features_spark = get_features(input_spark)
    features_pd = features_spark.toPandas()

    # ensure we don't have labels here
    if "is_ransomware" in features_pd.columns:
        features_pd = features_pd.drop(columns=["is_ransomware"])

    # align to training feature columns; missing columns → 0
    for col in FEATURE_COLUMNS:
        if col not in features_pd.columns:
            features_pd[col] = 0.0

    features_pd = features_pd[FEATURE_COLUMNS]

    proba = model.predict_proba(features_pd)[0, 1]

    return jsonify({"ransomware_probability": float(proba)}), 200
