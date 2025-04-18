import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from .utils import preprocess_data, save_model, detect_task
from .visualize import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, generate_pdf_report

class AutoML:
    def __init__(self, task='auto', random_state=42):
        """Initialize AutoML with task type and random state."""
        self.task = task
        self.random_state = random_state
        self.models = self._initialize_models()
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.results = {}
    
    def _initialize_models(self):
        """Initialize models based on task type."""
        if self.task == 'regression':
            return {
                'RandomForest': RandomForestRegressor(random_state=self.random_state),
                'XGBoost': xgb.XGBRegressor(random_state=self.random_state),
                'LightGBM': lgb.LGBMRegressor(random_state=self.random_state),
                'SVR': SVR()
            }
        else:
            return {
                'RandomForest': RandomForestClassifier(random_state=self.random_state),
                'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'XGBoost': xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False),
                'LightGBM': lgb.LGBMClassifier(random_state=self.random_state),
                'SVC': SVC(probability=True, random_state=self.random_state)
            }
    
    def load_data(self, file_path=None, dataframe=None, target_column=None):
        """Load data from CSV/Excel/JSON or DataFrame."""
        if file_path:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")
        elif dataframe is not None:
            self.data = dataframe
        else:
            raise ValueError("Provide either file_path or dataframe")
        
        if target_column:
            self.target = target_column
            if self.task == 'auto':
                self.task = detect_task(self.data[target_column])
        return self
    
    def preprocess(self, test_size=0.2, balance_classes=False):
        """Preprocess data and split into train/test."""
        if self.data is None or self.target is None:
            raise ValueError("Data or target column not loaded")
        
        X, y = preprocess_data(self.data, self.target, balance_classes=balance_classes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        return self
    
    def train(self, tune_hyperparams=False, save_path=None):
        """Train multiple models and select the best."""
        if self.X_train is None:
            raise ValueError("Data not preprocessed")
        
        self.results = {}
        for name, model in self.models.items():
            if tune_hyperparams:
                param_grid = self._get_param_grid(name)
                search = BayesSearchCV(model, param_grid, n_iter=10, cv=3, random_state=self.random_state)
                search.fit(self.X_train, self.y_train)
                self.models[name] = search.best_estimator_
                score = search.score(self.X_test, self.y_test)
            else:
                model.fit(self.X_train, self.y_train)
                score = model.score(self.X_test, self.y_test)
            
            self.results[name] = {'model': self.models[name], 'score': score}
            print(f"{name} {'Accuracy' if self.task != 'regression' else 'R2 Score'}: {score:.4f}")
        
        self.best_model = max(self.results, key=lambda x: self.results[x]['score'])
        print(f"Best Model: {self.best_model} with {'Accuracy' if self.task != 'regression' else 'R2 Score'}: {self.results[self.best_model]['score']:.4f}")
        
        if save_path:
            save_model(self.models[self.best_model], save_path)
        
        return self
    
    def _get_param_grid(self, model_name):
        """Define hyperparameter grid for tuning."""
        if model_name == 'RandomForest':
            return {'n_estimators': (50, 200), 'max_depth': (3, 20)}
        elif model_name in ['XGBoost', 'LightGBM']:
            return {'n_estimators': (50, 200), 'learning_rate': (0.01, 0.3, 'log-uniform'), 'max_depth': (3, 10)}
        elif model_name == 'SVC':
            return {'C': (0.1, 10, 'log-uniform'), 'kernel': ['rbf', 'linear']}
        return {}
    
    def evaluate(self, save_plot_dir=None, save_pdf=None):
        """Evaluate best model and generate reports."""
        if self.best_model is None:
            raise ValueError("No model trained")
        
        model = self.models[self.best_model]
        y_pred = model.predict(self.X_test)
        
        if self.task != 'regression':
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            if save_plot_dir:
                plot_confusion_matrix(self.y_test, y_pred, save_path=f"{save_plot_dir}/confusion_matrix.png")
                plot_roc_curve(model, self.X_test, self.y_test, save_path=f"{save_plot_dir}/roc_curve.png")
        else:
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R2 Score: {r2:.4f}")
        
        if save_plot_dir:
            plot_feature_importance(model, self.X_train, save_path=f"{save_plot_dir}/feature_importance.png")
        
        if save_pdf:
            generate_pdf_report(self.results, self.task, y_pred, self.y_test, save_pdf)
        
        return self
    
    def predict(self, new_data):
        """Make predictions with the best model."""
        if self.best_model is None:
            raise ValueError("No model trained")
        
        X_new = preprocess_data(new_data, target=None)
        return self.models[self.best_model].predict(X_new)
    
    def export_api(self, output_dir='api'):
        """Export best model as a FastAPI application."""
        from .api import generate_fastapi_app
        generate_fastapi_app(self.models[self.best_model], output_dir)
        print(f"FastAPI app generated in {output_dir}")