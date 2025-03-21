from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import time
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

# -------------------------------------------------------------------------------------
# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = os.getenv("REPO_OWNER")
repo_name = os.getenv("REPO_NAME")

# Set up MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# -------------------------------------------------------------------------------------
# Initialize Flask app
# -------------------------------------------------------------------------------------
app = Flask(__name__)

# Custom Metrics for Monitoring
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions", ["prediction"], registry=registry)

# -------------------------------------------------------------------------------------
# Load Model and Preprocessor
# -------------------------------------------------------------------------------------
model_name = "my_model"
preprocessor_name = "power_transformer"

def get_latest_model_version(model_name):
    """Fetch latest model version from MLflow"""
    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        return latest_version[0].version if latest_version else None
    except Exception as e:
        print(f"Error fetching model version: {e}")
        return None

model_version = get_latest_model_version(model_name)
if model_version:
    model_uri = f'models:/{model_name}/{model_version}'
    model = mlflow.pyfunc.load_model(model_uri)
else:
    model = None

# Load PowerTransformer
try:
    power_transformer = pickle.load(open('models/power_transformer.pkl', 'rb'))
except Exception as e:
    print(f"Error loading PowerTransformer: {e}")
    power_transformer = None

# -------------------------------------------------------------------------------------
# Feature Names
# -------------------------------------------------------------------------------------
feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# -------------------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    
    prediction = None
    input_values = [""] * len(feature_names)  # Empty placeholders for form

    if request.method == "POST":
        csv_input = request.form.get("csv_input", "").strip()
        
        if csv_input:
            try:
                # Convert CSV string into list of floats
                values = list(map(float, csv_input.split(",")))

                # Ensure correct number of features
                if len(values) != len(feature_names):
                    raise ValueError(f"Expected {len(feature_names)} values, but got {len(values)}")

                input_values = values  # Store values for UI
                
                # Convert to DataFrame
                features_df = pd.DataFrame([input_values], columns=feature_names)
                
                if power_transformer is None or model is None:
                    prediction = "Error: Model or Transformer not loaded properly."
                else:
                    # Apply PowerTransformer
                    transformed_features = power_transformer.transform(features_df)
                    transformed_df = pd.DataFrame(transformed_features, columns=features_df.columns)
                    
                    # Predict
                    result = model.predict(transformed_df)
                    prediction = "Fraud" if result[0] == 1 else "Non-Fraud"
                    
                    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
                
            except ValueError as ve:
                prediction = f"Input Error: {ve}"
            except Exception as e:
                prediction = f"Processing Error: {e}"
        
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return render_template("index.html", result=prediction, csv_input=",".join(map(str, input_values)))

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    
    csv_input = request.form.get("csv_input", "").strip()
    prediction = ""
    
    if csv_input:
        try:
            values = list(map(float, csv_input.split(",")))
            features_df = pd.DataFrame([values], columns=feature_names)
            
            if power_transformer is None or model is None:
                prediction = "Error: Model or Transformer not loaded properly."
            else:
                transformed_features = power_transformer.transform(features_df)
                transformed_df = pd.DataFrame(transformed_features, columns=features_df.columns)
                result = model.predict(transformed_df)
                prediction = "Fraud" if result[0] == 1 else "Non-Fraud"
                PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        except Exception as e:
            prediction = f"Error processing input: {e}"
    
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
    return prediction

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
