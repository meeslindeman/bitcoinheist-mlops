import pandas as pd
import numpy as np
import pytest

from src.model.model import Model


@pytest.fixture
def sample_data() -> pd.DataFrame:
    # note: small synthetic dataset for testing
    randomizer = np.random.RandomState(42)
    size = 40

    data = pd.DataFrame({
        "feature1": randomizer.rand(size),
        "feature2": randomizer.rand(size),
        "feature3": randomizer.rand(size),
        "is_ransomware": randomizer.randint(0, 2, size),
    })

    return data

def test_get_splits(sample_data):
    X_train, X_test, y_train, y_test = Model._get_splits(sample_data)

    # note: basic shape checks
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)

    # note: no overlap between train and test sets
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert set(y_train.index).isdisjoint(set(y_test.index))

def test_train_model(sample_data):
    model = Model()
    model.train_model(sample_data)

    assert model._model is not None
    assert model._cv_scores is not None
    assert "test_f1" in model._cv_scores

    summary = model.get_cv_scores()
    assert 0.0 <= summary["test_accuracy"].all() <= 1.0
    assert 0.0 <= summary["test_precision"].all() <= 1.0
    assert 0.0 <= summary["test_recall"].all() <= 1.0

def test_evaluate_model(sample_data):
    model = Model()
    model.train_model(sample_data)
    model.evaluate_model(sample_data)

    assert model._test_accuracy is not None
    assert model._test_report is not None
    assert model._test_roc_auc is not None

    assert 0.0 <= model._test_accuracy <= 1.0
    assert 0.0 <= model._test_roc_auc <= 1.0

def test_predict(sample_data):
    model = Model()
    model.train_model(sample_data)

    X = sample_data.drop(columns=["is_ransomware"]).iloc[:5]
    probabilities = model.predict_proba(X)

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.all() >= 0.0
    assert probabilities.all() <= 1.0
    assert probabilities.shape == (5, 2)
