"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    $ gunicorn --bind 0.0.0.0:8080 app:app
"""
import os
import json
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import pandas as pd
import joblib
from typing import Optional, Dict, Union, Tuple, List, Any
import numpy as np

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

MODEL = None
MODEL_INFO = {"workspace": "default", "model": "None", "version": "0.0"}

app = Flask(__name__)

# Configure Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def load_model(path: Path, info: Dict[str, str]) -> Tuple[Optional[Any], str]:
    """Helper function to safely load a model from disk."""
    try:
        model = joblib.load(path)
        log_message = f"Successfully loaded model: {info['model']} version {info['version']} from {path}"
        return model, log_message
    except Exception as e:
        log_message = f"ERROR: Failed to load model from {path}. Error: {e}"
        return None, log_message

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    try:
        with open(LOG_FILE, 'r') as f:
            log_content: List[str] = f.readlines()
    except FileNotFoundError:
        log_content = [f"ERROR: Log file not found at {LOG_FILE}"]
    except Exception as e:
        log_content = [f"ERROR: Failed to read log file: {e}"]
    return jsonify({"logs": log_content})

@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }

    """
    global MODEL, MODEL_INFO

    # 1. Parse Request
    try:
        json_data = request.get_json(force=True)
    except Exception:
        abort(400, description="Invalid JSON format.")

    required_fields = ["workspace", "model", "version"]
    if not all(field in json_data for field in required_fields):
        return jsonify({"status": "failed", "message": "Missing required fields"}), 400

    workspace = json_data.get("workspace", "")
    model_name = json_data.get("model", "")
    version = json_data.get("version", "")

    app.logger.info(f"Request received to load model: {model_name}")

    # 2. Find the Model File
    model_path = None

    # STRATEGY A: Check Root Directory (Local .pkl files)
    # This is what you need right now since your files are next to app.py
    local_root_file = Path(f"{model_name}.pkl")
    if local_root_file.exists():
        app.logger.info(f"Found model file locally in root: {local_root_file}")
        model_path = local_root_file

    # STRATEGY B: Check Registry Cache (Comet/WandB folder structure)
    # else:
    #     app.logger.info(f"Model {model_name} not found in root. Checking Registry Cache...")
    #     model_path = RegistryClient.check_local_model(workspace, model_name, version)
    #
    #     # STRATEGY C: Download (Simulated for now)
    #     if model_path is None:
    #          model_path = RegistryClient.download_model(workspace, model_name, version)

    # 3. Load the Model
    response = {"status": "failed", "message": "Model not found."}

    if model_path and model_path.exists():
        new_model, log_msg = load_model(model_path, json_data)
        app.logger.info(log_msg)

        if new_model:
            MODEL = new_model
            MODEL_INFO.update(json_data)
            response = {
                "status": "success",
                "message": f"Model {model_name} version {version} loaded successfully."
            }
    else:
        app.logger.warning(f"Could not find or download model: {model_name}")
        response["message"] = f"Failed to find model {model_name}. Keeping current model."

    return jsonify(response)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests to predict goal probability.
    """
    global MODEL

    if MODEL is None:
        app.logger.error("Prediction attempted without loaded model.")
        return jsonify({"status": "error", "message": "No model loaded. Call /download_registry_model first."}), 503

    # 1. Parse Data
    try:
        json_data = request.get_json(force=True)
        # Note: orient='columns' expects {"col1": {0: val, 1: val}, "col2": ...}
        input_df = pd.read_json(json.dumps(json_data), orient='columns')
    except Exception as e:
        app.logger.error(f"Data parsing failed: {e}")
        return jsonify({"status": "error", "message": "Invalid Data Format. Use df.to_json(orient='columns')."}), 400

    # 2. Feature Selection
    # Default to distance/angle if model name isn't specific, or check columns
    features = ['distanceToNet', 'shotAngle']

    # Check if features exist in input
    if not all(col in input_df.columns for col in features):
         return jsonify({"status": "error", "message": f"Input data must contain columns: {features}"}), 400

    # 3. Predict
    try:
        X = input_df[features]
        predictions = MODEL.predict_proba(X)[:, 1]

        response_df = pd.DataFrame({"expected_goal_prob": predictions.round(4)})
        return jsonify({"predictions": response_df.to_json(orient='records')})

    except Exception as e:
        app.logger.error(f"Prediction logic failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Running in debug mode for development
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)