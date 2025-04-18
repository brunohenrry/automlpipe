import os
import joblib

def generate_fastapi_app(model, output_dir):
    """Generate FastAPI app for model deployment."""
    os.makedirs(output_dir, exist_ok=True)
    
    app_code = """
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load model
model = joblib.load('model.pkl')

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    X = preprocess_data(df)
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction) if isinstance(prediction, np.integer) else float(prediction)}

def preprocess_data(df):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    scaler = StandardScaler()
    return scaler.fit_transform(df)
"""
    
    with open(f"{output_dir}/app.py", 'w') as f:
        f.write(app_code)
    
    joblib.dump(model, f"{output_dir}/model.pkl")