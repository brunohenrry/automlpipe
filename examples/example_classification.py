from automlpipe import AutoML
import pandas as pd

# Simulated iris dataset
df = pd.DataFrame({
    'sepal_length': [5.1, 4.9, 7.0, 6.4],
    'sepal_width': [3.5, 3.0, 3.2, 3.2],
    'petal_length': [1.4, 1.4, 4.7, 4.5],
    'petal_width': [0.2, 0.2, 1.4, 1.5],
    'species': ['setosa', 'setosa', 'versicolor', 'versicolor']
})

# Run AutoML
aml = AutoML(task='classification')
aml.load_data(dataframe=df, target_column='species').preprocess(balance_classes=True)
aml.train(tune_hyperparams=True, save_path="iris_model.pkl")
aml.evaluate(save_plot_dir="iris_plots", save_pdf="iris_report.pdf")