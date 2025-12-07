import os
import pickle
import tempfile
import numpy as np
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, roc_auc_score

from configs.configs import ModelConfig, PathsConfig, RunConfig


class Model:
    def __init__(self):
        self._model = None
        self._cv_scores: dict | None = None
        self._test_report: dict | None = None
        self._test_roc_auc: float | None = None
    
    @staticmethod
    def _get_model_object() -> RandomForestClassifier:
        # note: classes are balanced in preprocessing step so no need for class_weight
        return RandomForestClassifier(
            n_estimators=ModelConfig.n_estimators,
            max_depth=ModelConfig.max_depth, 
            random_state=ModelConfig.random_state,
            n_jobs=-1,
        )
    
    @staticmethod
    def _get_splits(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = data.drop(columns=["is_ransomware"])
        y = data["is_ransomware"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=RunConfig.test_size, stratify=y, random_state=RunConfig.random_seed
        )

        return X_train, X_test, y_train, y_test
    
    def _ensure_trained(self) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been trained/loaded yet.")
    
    def train_model(self, data: pd.DataFrame) -> None:
        X_train, X_test, y_train, y_test = self._get_splits(data)

        classifier = self._get_model_object()
        
        self._cv_scores = cross_validate(
            classifier,
            X_train,
            y_train,
            cv=RunConfig.num_folds,
            return_train_score=True,
            n_jobs=-1,
            scoring=["precision", "recall", "f1"]
        )

        self._model = classifier.fit(X_train, y_train)

        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)[:, 1]

        self._test_report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        self._test_roc_auc = roc_auc_score(y_test, y_proba)

    def get_cv_scores(self) -> dict:
        if self._cv_scores is None:
            raise RuntimeError("Model has not been trained yet.")
        return self._cv_scores
    
    def evaluate_model(self, data: pd.DataFrame) -> None:
        if self._test_report is not None and self._test_roc_auc is not None:
            print("Classification report (test set):")
            print(pd.DataFrame(self._test_report).T.to_string())
            print(f"\nROC AUC Score (test set): {self._test_roc_auc:.6f}")
            return

        if data is None:
            raise RuntimeError("No stored test metrics and no data provided to recompute.")

        self._ensure_trained()
        X_train, X_test, y_train, y_test = self._get_splits(data)
        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba)}")

    def save_model_local(self) -> None:
        self._ensure_trained()
        path = os.path.join(PathsConfig.model_path, f"{ModelConfig.model_name}.pkl")
        os.makedirs(PathsConfig.model_path, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        print(f"Model saved to {path}")

    def load_model_local(self) -> None:
        path = os.path.join(PathsConfig.model_path, f"{ModelConfig.model_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found at {path}. Did you run training + save_model_local()?"
            )

        with open(path, "rb") as f:
            self._model = pickle.load(f)
        print(f"Model loaded from {path}")

    def log_model_to_mlflow(self) -> None:
        self._ensure_trained()
        if self._cv_scores is None:
            raise RuntimeError("Model has not been trained yet (no CV scores).")

        mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)

        experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(RunConfig.experiment_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id, run_name=RunConfig.run_name) as run:
            run_id = run.info.run_id

            # log CV scores
            mlflow.log_dict(self._cv_scores, "cv_scores.json")
            mlflow.log_metric("cv_test_f1_mean", self._cv_scores["test_f1"].mean())
            mlflow.log_metric("cv_test_f1_std", self._cv_scores["test_f1"].std())
            mlflow.log_metric("cv_test_precision_mean", self._cv_scores["test_precision"].mean())
            mlflow.log_metric("cv_test_recall_mean", self._cv_scores["test_recall"].mean())

            # log test metrics if available
            if self._test_roc_auc is not None:
                mlflow.log_metric("test_roc_auc", float(self._test_roc_auc))

            if self._test_report is not None:
                mlflow.log_dict(self._test_report, "test_classification_report.json")

            # log model artifact under a stable subdirectory (RunConfig.run_name)
            with tempfile.NamedTemporaryFile("wb", delete=False) as temp_file:
                tmp_path = temp_file.name
            model_file_name = f"{ModelConfig.model_name}.pkl"
            with open(tmp_path, "wb") as f:
                pickle.dump(self._model, f)
            mlflow.log_artifact(tmp_path, artifact_path=RunConfig.run_name)
            os.remove(tmp_path)

            return run_id
        
    def load_model_from_mlflow(self, run_id: str) -> None:
        mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)

        experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
        if experiment is None:
            raise RuntimeError(f"Experiment {RunConfig.experiment_name} does not exist in MLflow.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=RunConfig.run_name,
                dst_path=tmp_dir,
            )
            model_file_name = f"{ModelConfig.model_name}.pkl"
            artifacts_path = os.path.join(tmp_dir, RunConfig.run_name)
            model_path = os.path.join(artifacts_path, model_file_name)

            if not os.path.exists(model_path):
                raise RuntimeError(f"Model file {model_file_name} not found under MLflow artifacts.")

            with open(model_path, "rb") as f:
                self._model = pickle.load(f)

    def predict(self, features: pd.DataFrame):
        self._ensure_trained()
        return self._model.predict(features)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self._ensure_trained()
        return self._model.predict_proba(features)