import pandas as pd
import pickle

from configs.configs import ModelConfig, PathsConfig, RunConfig

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, roc_auc_score

class Model:
    def __init__(self):
        self.model = None
    
    @staticmethod
    def _get_model_object() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=ModelConfig.n_estimators,
            max_depth=ModelConfig.max_depth, 
            class_weight=ModelConfig.class_weight,
            random_state=ModelConfig.random_state
        )
    
    @staticmethod
    def _get_splits(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = data.drop(columns=["is_ransomware"])
        y = data["is_ransomware"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=RunConfig.test_size, stratify=y, random_state=RunConfig.random_seed
        )

        return X_train, X_test, y_train, y_test
    
    def train_model(self, data: pd.DataFrame) -> None:
        X_train, X_test, y_train, y_test = self._get_splits(data)
        classifier = self._get_model_object()
        self.model = classifier.fit(X_train, y_train)

    def evaluate_model(self, data: pd.DataFrame) -> None:
        X_train, X_test, y_train, y_test = self._get_splits(data)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba)}")

    def get_cv_scores(self, data: pd.DataFrame) -> dict:
        X_train, _, y_train, _ = self._get_splits(data)
        classifier = self._get_model_object()
        cv_results = cross_validate(
            classifier,
            X_train,
            y_train,
            cv=RunConfig.num_folds,
            return_train_score=True,
            n_jobs=-1,
            scoring=["precision", "recall", "f1"],
        )
        return cv_results
    
    def save_model(self):
        with open(f"{PathsConfig.model_path}{ModelConfig.model_name}.pkl", "wb") as f:
            pickle.dump(self.model, f)