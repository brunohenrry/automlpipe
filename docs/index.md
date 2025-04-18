AutoMLPipe Documentation
Overview
AutoMLPipe automates Machine Learning pipelines, from data ingestion to model deployment, with simple APIs and a CLI.
Installation
pip install automlpipe

Quick Start
from automlpipe import AutoML

aml = AutoML(task='classification')
aml.load_data("data.csv", target_column="target").preprocess(balance_classes=True)
aml.train(tune_hyperparams=True, save_path="model.pkl")
aml.evaluate(save_plot_dir="plots", save_pdf="report.pdf")
aml.export_api(output_dir="api")

CLI Usage
automlpipe --data data.csv --target target --task classification --output results

API Reference

AutoML(task='auto', random_state=42): Initialize with task type.
load_data(file_path=None, dataframe=None, target_column=None): Load data.
preprocess(test_size=0.2, balance_classes=False): Preprocess and split data.
train(tune_hyperparams=False, save_path=None): Train models.
evaluate(save_plot_dir=None, save_pdf=None): Generate reports.
predict(new_data): Make predictions.
export_api(output_dir='api'): Export as FastAPI app.

Tutorials
See tutorial.md for detailed examples.
