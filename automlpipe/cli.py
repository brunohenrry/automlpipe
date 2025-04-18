import click
from .core import AutoML

@click.command()
@click.option('--data', help='Path to data file (CSV/Excel/JSON)')
@click.option('--target', help='Target column name')
@click.option('--task', default='auto', help='Task type: auto, classification, regression')
@click.option('--output', default='output', help='Output directory for results')
def main(data, target, task, output):
    """AutoMLPipe CLI for automated ML pipelines."""
    aml = AutoML(task=task)
    aml.load_data(file_path=data, target_column=target).preprocess().train(tune_hyperparams=True, save_path=f"{output}/model.pkl")
    aml.evaluate(save_plot_dir=output, save_pdf=f"{output}/report.pdf")
    aml.export_api(output_dir=output)
    print(f"Results saved in {output}")