import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.model import train_model
from ml.data import process_data
import pandas as pd
import os


def test_train_model():
    """
    Tests to ensure that the the model is Random Forest and passed random state is 42
    """
    X = np.array([[6,8], [5,5], [6,7]])
    y = np.array([1,0,1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == 42


def test_data_split():
    """
    Tests to check if the data was split as intended 80/20
    """
    data = pd.read_csv(os.path.join('data', 'census.csv'))

    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data['salary']
    )

    ex_test = math.ceil(len(data) * 0.2)
    ex_train = len(data) - ex_test

    assert len(test) == ex_test
    assert len(train) == ex_train


def test_three():
    """
    Tests to see if labels in the dataset are properly binarized
    """
    data = pd.read_csv(os.path.join('data', 'census.csv'))

    cat_features = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race",
        "sex", "native-country",
    ]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    unique_values = set(y)

    assert unique_values.issubset({0, 1})
