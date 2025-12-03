from pyspark.sql import DataFrame

from src.features import get_features as _get_features


class FeatureExtractor:
    def __init__(self):
        pass

    def get_features(self, data: DataFrame) -> DataFrame:
        return _get_features(data)
    
    def save_to_mlflow(self, run_id: str):
        # Placeholder for saving feature extractor to MLflow
        return None
    
    def load_from_mlflow(self, run_id: str):
        # Placeholder for loading feature extractor from MLflow
        return None