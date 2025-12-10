from src.model import Model
import pandas as pd
import numpy as np

def sample_data() -> pd.DataFrame:
    # note: small synthetic dataset for testing
    randomizer = np.random.RandomState(42)
    size = 20

    data = pd.DataFrame({
        "feature1": randomizer.rand(size),
        "feature2": randomizer.rand(size),
        "feature3": randomizer.rand(size),
        "is_ransomware": randomizer.randint(0, 2, size),
    })

    return data

model = Model()
data = sample_data()
model.train_model(data)
summary = model.get_cv_scores()
print("Cross-validation scores summary:")
print(summary)
print("================================")

def test_get_splits(small_data):
    X_train, X_test, y_train, y_test = Model._get_splits(small_data)

    # note: basic shape checks
    assert len(X_train) + len(X_test) == len(small_data)
    assert len(y_train) + len(y_test) == len(small_data)

    # note: no overlap between train and test sets
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert set(y_train.index).isdisjoint(set(y_test.index))

test_get_splits(sample_data())