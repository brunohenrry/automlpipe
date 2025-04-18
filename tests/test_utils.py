import pytest
import pandas as pd
from automlpipe.utils import preprocess_data, detect_task

def test_preprocess_data():
    df = pd.DataFrame({'A': [1, None, 3], 'B': ['x', 'y', 'z'], 'target': [0, 1, 0]})
    X, y = preprocess_data(df, 'target')
    assert X.shape == (3, 2)
    assert not np.any(np.isnan(X))

def test_detect_task():
    s = pd.Series([0, 1, 0])
    assert detect_task(s) == 'classification'
    s = pd.Series([1.2, 3.4, 5.6])
    assert detect_task(s) == 'regression'