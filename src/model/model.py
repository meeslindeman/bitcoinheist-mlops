import numpy as np
import pandas as pd
import logging

from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_validate, train_test_split

from configs.configs import ModelConfig, RunConfig
from src.utils.mlflow_utils import log_model_to_mlflow, load_model_from_mlflow


logger = logging.getLogger(__name__)    


class Model:
    def __init__(self):
        self._model: BaseEstimator | None = None
        self._cv_scores: dict | None = None
        self._test_accuracy: float | None = None
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
            scoring=["accuracy", "precision", "recall", "f1"]
        )

        self._model = classifier.fit(X_train, y_train)

        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)[:, 1]

        self._test_accuracy = accuracy_score(y_test, y_pred)
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
    
    # note: small utility to evaluate model and print classification report and ROC AUC for debugging purposes
    def evaluate_model(self, data: pd.DataFrame | None = None) -> None:
        # note: if test metrics are already stored, print them directly
        if self._test_report is not None and self._test_roc_auc is not None:
            logger.info("Classification report (test set):")
            logger.info(pd.DataFrame(self._test_report).T.to_string())
            logger.info(f"\nROC AUC Score (test set): {self._test_roc_auc:.6f}")
            return

        if data is None:
            raise RuntimeError("No stored test metrics and no data provided to recompute.")

        # note: recompute test metrics
        self._ensure_trained()
        X_train, X_test, y_train, y_test = self._get_splits(data)
        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)[:, 1]

        logger.info(classification_report(y_test, y_pred))
        logger.info(f"ROC AUC Score: {roc_auc_score(y_test, y_proba)}")

    # note: get a summary of test metrics for telemetry/logging purposes
    def get_test_summary(self) -> dict:
        if self._test_report is None or self._test_roc_auc is None:
            raise RuntimeError("Test metrics have not been computed yet.")

        positive_label = "1"
        f1_positive = None
        if positive_label in self._test_report:
            f1_positive = self._test_report[positive_label].get("f1-score")

        summary: dict = {
            "accuracy": float(self._test_accuracy) if self._test_accuracy is not None else None,
            "roc_auc": float(self._test_roc_auc),
            "f1_positive": float(f1_positive) if f1_positive is not None else None
        }

        if self._cv_scores is not None and "test_f1" in self._cv_scores:
            summary["cv_test_f1_mean"] = float(self._cv_scores["test_f1"].mean())
            summary["cv_test_f1_std"] = float(self._cv_scores["test_f1"].std())

        return summary

    def log_model_to_mlflow(self) -> str:
        self._ensure_trained()
        if self._cv_scores is None:
            raise RuntimeError("Model has not been trained yet (no CV scores).")

        return log_model_to_mlflow(
            model=self._model,
            cv_scores=self._cv_scores,
            test_report=self._test_report,
            test_roc_auc=self._test_roc_auc,
            test_accuracy=self._test_accuracy
        )

    # note: load model from MLflow and set as the current model
    def load_model_from_mlflow(self, run_id: str) -> None:
        model = load_model_from_mlflow(run_id=run_id)
        self._model = model

    # note: only predict probabilities and convert to labels later
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self._ensure_trained()
        return self._model.predict_proba(features)