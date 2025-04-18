import pytest
import pandas as pd
from automlpipe import AutoML

def test_load_data():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'target': [0, 1, 0]})
    aml = AutoML().load_data(dataframe=df, target_column='target')
    assert aml.data.shape == (3, 3)
    assert aml.task == 'classification'

def test_preprocess():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'target': [0, 1, 0]})
    aml = AutoML().load_data(dataframe=df, target_column='target').preprocess()
    assert aml.X_train.shape[1] == 2

def test_train():
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['x', 'y', 'x', 'y'], 'target': [0, 1, 0, 1]})
    aml = AutoML().load_data(dataframe=df, target_column='target').preprocess().train()
    assert aml.best_model in aml.models