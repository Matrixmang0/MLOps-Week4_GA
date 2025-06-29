import joblib
import pytest
import numpy as np
import pytest
from feast import FeatureStore
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

MODEL_PATH = "./artifacts/model.joblib"
ACCURACY_THRESHOLD = 0.9

@pytest.fixture(scope="module")
def iris_data():
    """Load and split the iris dataset."""
    iris = pd.read_csv("./data/iris.csv")
    features = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    target = iris["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test

@pytest.fixture(scope="module")
def loaded_model():
    """Load the model from joblib."""
    return joblib.load(MODEL_PATH)

def test_model_instance(loaded_model):
    """Check that the loaded model is a sklearn classifier."""
    assert isinstance(loaded_model, ClassifierMixin), "Loaded model is not a classifier."

def test_model_not_null(loaded_model):
    """Check model is not None."""
    assert loaded_model is not None, "Model is None"

def test_model_predict_shape(loaded_model, iris_data):
    """Ensure model.predict returns expected number of outputs."""
    _, X_test, _, y_test = iris_data
    y_pred = loaded_model.predict(X_test)
    assert len(y_pred) == len(y_test), "Mismatch in prediction and test set length"

def test_model_accuracy(loaded_model, iris_data):
    """Check model accuracy."""
    _, X_test, _, y_test = iris_data
    y_pred = loaded_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    assert acc >= ACCURACY_THRESHOLD, f"Accuracy {acc} below threshold {ACCURACY_THRESHOLD}"

def test_model_precision_recall_f1(loaded_model, iris_data):
    """Evaluate precision, recall, and F1-score."""
    _, X_test, _, y_test = iris_data
    y_pred = loaded_model.predict(X_test)
    for avg in ['macro', 'micro', 'weighted']:
        precision = precision_score(y_test, y_pred, average=avg)
        recall = recall_score(y_test, y_pred, average=avg)
        f1 = f1_score(y_test, y_pred, average=avg)
        assert 0 <= precision <= 1, f"Invalid precision: {precision}"
        assert 0 <= recall <= 1, f"Invalid recall: {recall}"
        assert 0 <= f1 <= 1, f"Invalid F1 score: {f1}"

def test_model_predict_values(loaded_model, iris_data):
    """Ensure predictions are within valid class range."""
    _, X_test, _, y_test = iris_data
    y_pred = loaded_model.predict(X_test)
    valid_classes = set(np.unique(y_test))
    pred_classes = set(np.unique(y_pred))
    assert pred_classes.issubset(valid_classes), "Model predicted invalid class labels"

@pytest.fixture(scope="module")
def store():
    """Return a Feast FeatureStore object pointing to the repo."""
    return FeatureStore(repo_path="./iris_feature_store/feature_repo")

def test_registry_connection(store):
    feature_views = store.list_feature_views()
    assert any(fv.name == "iris_features" for fv in feature_views), \
        "'iris_features' not found in feature views"

def test_online_feature_retrieval(store):
    entity_rows = [{"species_id": 0}]
    features = store.get_online_features(
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
            "iris_features:species"
        ],
        entity_rows=entity_rows
    ).to_dict()

    for key in features:
        assert features[key][0] is not None, f"{key} is None"