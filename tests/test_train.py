import json
import os
import pickle
import pytest
from sklearn.datasets import load_digits

def test_config_load():
    with open('config/config.json') as f:
        config = json.load(f)
    assert 'C' in config and isinstance(config['C'], float)
    assert 'solver' in config and isinstance(config['solver'], str)
    assert 'max_iter' in config and isinstance(config['max_iter'], int)

def test_model_fitting():
    with open('config/config.json') as f:
        config = json.load(f)
    from sklearn.linear_model import LogisticRegression
    data = load_digits()
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=42
    )
    model.fit(data.data, data.target)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'classes_')

def test_model_accuracy():
    with open('config/config.json') as f:
        config = json.load(f)
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    digits = load_digits()
    X, y = digits.data, digits.target
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=42
    )
    model.fit(X, y)
    acc = model.score(X, y)
    assert acc > 0.9
