AutoMLPipe Tutorial
Classification Example
from automlpipe import AutoML
import pandas as pd

# Load data
aml = AutoML(task='classification')
aml.load_data(file_path="iris.csv", target_column="species")

# Preprocess with class balancing
aml.preprocess(balance_classes=True)

# Train with hyperparameter tuning
aml.train(tune_hyperparams=True, save_path="iris_model.pkl")

# Evaluate
aml.evaluate(save_plot_dir="iris_plots", save_pdf="iris_report.pdf")

# Predict
new_data = pd.read_csv("iris_new.csv")
predictions = aml.predict(new_data)
print("Predictions:", predictions)

Regression Example
from automlpipe import AutoML
import pandas as pd

# Load data
aml = AutoML(task='regression')
aml.load_data(file_path="housing.csv", target_column="price")

# Preprocess
aml.preprocess()

# Train
aml.train(save_path="housing_model.pkl")

# Evaluate
aml.evaluate(save_plot_dir="housing_plots")

# Export API
aml.export_api(output_dir="housing_api")

CLI Example
automlpipe --data iris.csv --target species --task classification --output iris_results

