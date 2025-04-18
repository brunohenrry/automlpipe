import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import dask.dataframe as dd

def preprocess_data(data, target=None, balance_classes=False):
    """Preprocess data for ML."""
    # Convert to Dask for large datasets
    if isinstance(data, pd.DataFrame) and data.memory_usage().sum() > 1e9:  # 1GB
        df = dd.from_pandas(data, npartitions=4)
    else:
        df = data.copy()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    
    # Split features and target
    if target:
        X = df.drop(columns=[target])
        y = df[target]
        if y.dtype == 'object':
            y = le.fit_transform(y)
    else:
        X = df
        y = None
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X.compute() if isinstance(X, dd.DataFrame) else X)
    
    # Balance classes if requested
    if balance_classes and target and y is not None:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    
    return X, y if target else X

def save_model(model, path):
    """Save model to disk."""
    joblib.dump(model, path)

def detect_task(target_series):
    """Detect if task is classification or regression."""
    if target_series.dtype in ['int64', 'object'] or len(target_series.unique()) < 20:
        return 'classification'
    return 'regression'