import pickle
import tempfile
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

        report = classification_report(
            y_test, 
            y_pred, 
            output_dict=True,
            zero_division=0
        )
        roc_auc = roc_auc_score(y_test, y_proba)

        self._test_report = report
        self._test_roc_auc = roc_auc

    def get_cv_scores(self, data: pd.DataFrame) -> dict:
        if self._cv_scores is None:
            raise RuntimeError("Model has not been trained yet.")
        return self._cv_scores
    
    def evaluate_model(self, data: pd.DataFrame) -> None:
        if self._test_report is None or self._test_roc_auc is None:
            # note: fall back to recomputing if needed
            X_train, X_test, y_train, y_test = self._get_splits(data)
            y_pred = self._model.predict(X_test)
            y_proba = self._model.predict_proba(X_test)[:, 1]

            print(classification_report(y_test, y_pred))
            print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba)}")
            return

        print("Classification report (test set):")
        print(pd.DataFrame(self._test_report).T.to_string())
        print(f"\nROC AUC Score (test set): {self._test_roc_auc:.6f}")

    def save_model(self) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        
        path = f"{PathsConfig.model_path}{ModelConfig.model_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        print(f"Model saved to {path}")

    def log_modelto_mlflow(self) -> None:
        if self._model is None or self._cv_scores is None:
            raise RuntimeError("Model has not been trained or CV scores missing.")
        
        mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)

        if experiment := mlflow.get_experiment_by_name(RunConfig.experiment_name):
            mlflow_experiment_id = experiment.experiment_id
        else:
            mlflow_experiment_id = mlflow.create_experiment(RunConfig.experiment_name)

        with mlflow.start_run(experiment_id=mlflow_experiment_id):
            # note: log CV scores
            mlflow.log_dict(self._cv_scores, "cv_scores.json")
            mlflow.log_metric("cv_test_f1_mean", self._cv_scores["test_f1"].mean())
            mlflow.log_metric("cv_test_f1_std", self._cv_scores["test_f1"].std())
            mlflow.log_metric(
                "cv_test_precision_mean", self._cv_scores["test_precision"].mean()
            )
            mlflow.log_metric(
                "cv_test_recall_mean", self._cv_scores["test_recall"].mean()
            )

            # note: optional log test ROC AUC if available
            if self._test_roc_auc is not None:
                mlflow.log_metric("test_roc_auc", float(self._test_roc_auc))

            # note: log model artifact
            with tempfile.NamedTemporaryFile("wb") as temp_file:
                tmp_path = temp_file.name + ".pkl"
                with open(tmp_path, "wb") as f:
                    pickle.dump(self._model, f)
                mlflow.log_artifact(tmp_path, f"{ModelConfig.model_name}.pkl")