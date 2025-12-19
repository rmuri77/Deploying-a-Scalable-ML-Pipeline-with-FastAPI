import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

# Keep these aligned with train_model.py
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


@pytest.fixture(scope="module")
def data():
    """Load a small sample of census data for tests."""
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    df = pd.read_csv(data_path)

    # Small sample keeps tests fast (but still real data)
    return df.sample(n=1000, random_state=42)


def test_process_data_shapes(data):
    """process_data should return X, y with matching rows and a fitted encoder/lb."""
    X, y, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == data.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model_returns_logistic_regression(data):
    """train_model should return a LogisticRegression model (per project)."""
    X, y, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )
    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)


def test_inference_and_metrics_valid_range(data):
    """inference outputs correct length; metrics are floats in [0, 1]."""
    train_df = data.sample(frac=0.8, random_state=1)
    test_df = data.drop(train_df.index)

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert len(preds) == len(y_test)

    p, r, f1 = compute_model_metrics(y_test, preds)
    for v in (p, r, f1):
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
