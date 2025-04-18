from automlpipe import AutoML
import pandas as pd

# Simulated housing dataset
df = pd.DataFrame({
    'size': [1500, 2000, 1800, 2500],
    'rooms': [3, 4, 3, 5],
    'price': [300000, 400000, 350000, 500000]
})

# Run AutoML
aml = AutoML(task='regression')
aml.load_data(dataframe=df, target_column='price').preprocess()
aml.train(save_path="housing_model.pkl")
aml.evaluate(save_plot_dir="housing_plots")
aml.export_api(output_dir="housing_api")