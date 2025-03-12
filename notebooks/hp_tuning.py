import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = "https://dagshub.com/AyanArshad02/Credit-Fraud-Detection.mlflow"
dagshub.init(repo_owner='AyanArshad02', repo_name='Credit-Fraud-Detection', mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Logistic Regression with PowerTransformer")

# ==========================
# Load & Prepare Data
# ==========================
def load_and_prepare_data(filepath):
    """Loads dataset, applies PowerTransformer, and splits data."""
    df = pd.read_csv(filepath)
    
    # Split features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # Apply PowerTransformer to normalize features
    transformer = PowerTransformer()
    X_transformed = transformer.fit_transform(X)
    
    return train_test_split(X_transformed, y, test_size=0.2, random_state=42), transformer

# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, transformer):
    """Trains Logistic Regression with GridSearch and logs results to MLflow."""
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    
    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        # Log parameters & metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "logistic_regression_model")
        
        print(f"Best Params: {grid_search.best_params_} | F1 Score: {metrics['f1_score']:.4f}")

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), transformer = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, transformer)

