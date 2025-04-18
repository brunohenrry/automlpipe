# AutoMLPipe

A Python library for automated Machine Learning pipelines, simplifying data preprocessing, model training, evaluation, and deployment.

## Installation

```bash
pip install automlpipe
```

## Quick Start

```python
from automlpipe import AutoML

# Load and preprocess data
aml = AutoML(task='classification')
aml.load_data(file_path="data.csv", target_column="target").preprocess(balance_classes=True)

# Train models with hyperparameter tuning
aml.train(tune_hyperparams=True, save_path="best_model.pkl")

# Evaluate and generate reports
aml.evaluate(save_plot_dir="plots", save_pdf="report.pdf")

# Export as API
aml.export_api(output_dir="api")

# Predict on new data
new_data = pd.read_csv("new_data.csv")
predictions = aml.predict(new_data)
print("Predictions:", predictions)
```

## CLI Usage

```bash
automlpipe --data data.csv --target target --task classification --output results
```

## Features
- Flexible data ingestion (CSV, Excel, JSON, pandas DataFrames).
- Automated preprocessing (missing values, encoding, scaling, class balancing).
- Multi-model training (Random Forest, XGBoost, LightGBM, SVM, Logistic Regression).
- Hyperparameter tuning with Bayesian optimization.
- Comprehensive evaluation with metrics and visualizations.
- Model export to joblib, ONNX, or FastAPI app.
- PDF report generation.
- Scalable with Dask for large datasets.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, lightgbm, skopt, matplotlib, seaborn, imbalanced-learn, fastapi, uvicorn, reportlab, dask, click

## Documentation
See [docs/index.md](docs/index.md) for full API reference and tutorials.

## Examples
Check the [examples/](examples/) directory for sample scripts:
- `example_classification.py`
- `example_regression.py`