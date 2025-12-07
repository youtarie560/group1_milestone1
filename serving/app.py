import os
import json
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import pandas as pd
import joblib
from typing import Optional, Dict, Union, Tuple, List, Any
import wandb

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
from dotenv import load_dotenv

load_dotenv()

MODEL = None
MODEL_INFO = {"workspace": "default",
              "model": "None",
              "version": "0.0"}
API="fbe2d0aad81c2ed1f8e2852a2c48e6f74bae90b5"
app = Flask(__name__)

# Configure Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def load_model(path: Path, info: Dict[str, str]) -> Tuple[Optional[Any], str]:
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
        log_content = [f"[ERROR] Log file not found at {LOG_FILE}"]
    except Exception as e:
        log_content = [f"[ERROR] Failed to read log file: {e}"]
    return jsonify({"logs": log_content})


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests to change the model.
    STRICT MODE: Only allows 'Distance' and 'DistanceAngle'.
    """
    global MODEL, MODEL_INFO
    current_model_name = MODEL_INFO.get("model", "None")

    try:
        json_data = request.get_json(force=True)
    except Exception:
        abort(400, description="Invalid JSON format.")

    required_fields = ["workspace", "model", "version"]
    if not all(field in json_data for field in required_fields):
        return jsonify({"status": "failed", "message": "Missing required fields"}), 400

    workspace = json_data.get("workspace", "")
    model_name = json_data.get("model", "")

    app.logger.info(f"Request received to load model: {model_name}")

    valid_models = {
        "RL-Distance": {
            "run_name": "LogisticRegression_Logistic Regression (Distance)",
            "filename": "Logistic Regression (Distance)_logreg_model.pkl"
        },
        "RL-Angle":{
            "run_name": "LogisticRegression_Logistic Regression (Angle)",
            "filename": "Logistic Regression (Angle)_logreg_model.pkl"
        },
        "RL-DistanceAngle": {
            "run_name": "LogisticRegression_Logistic Regression (Distance+Angle)",
            "filename": "Logistic Regression (Distance+Angle)_logreg_model.pkl"
        }
    }

    if model_name not in valid_models:
        msg = f"Invalid model '{model_name}'. Current model set back to {current_model_name}."
        app.logger.warning(msg)
        return jsonify({
            "status": "failed",
            "message": f"Invalid model '{model_name}'. Allowed models: 'Distance', 'DistanceAngle'"
        }), 400

    # Retrieve the specific config for the valid model
    target_config = valid_models[model_name]
    run_name = target_config["run_name"]
    expected_filename = target_config["filename"]
    model_path = Path(expected_filename)

    if model_path.exists():
        app.logger.info(f"Model file {model_path} found locally.")

        new_model, log_msg = load_model(model_path, json_data)

        if new_model:
            MODEL = new_model
            MODEL_INFO.update(json_data)
            app.logger.info(f"Model successfully changed from {current_model_name} to {model_name}.")
            return jsonify({
                "status": "success",
                "message": f"Model {model_name} found locally and loaded successfully."
            })
        else:
            app.logger.error(f"File exists but failed to load. Current model set back to {current_model_name}.")
            return jsonify({"status": "failed", "message": "File exists locally but failed to load."}), 500

    # ---------------------------------------------------------
    # STEP 3: Download from WandB (If not found locally)
    # ---------------------------------------------------------
    app.logger.info(f"Model {model_name} not found locally. Attempting download from WandB...")

    try:
        key = API
        if not key:
            raise ValueError("API_KEY not set.")

        wandb.login(key=key)
        api = wandb.Api()
        ENTITY = "IFT67582025A1"
        PROJECT = "IFT6758.2025-A01"
        path_to_check = f"{ENTITY}/{PROJECT}"

        runs = api.runs(path_to_check)
        run = next((r for r in runs if r.name == run_name), None)

        if not run:
            raise Exception(f"Run '{run_name}' not found in project '{path_to_check}'")

        run.file(expected_filename).download(root=".", replace=True)

        app.logger.info(f"Successfully downloaded {expected_filename} from WandB.")

    except Exception as e:
        #return
        print(f"WandB download failed for '{model_name}'. Current model set back to {current_model_name}. Error: {str(e)}")
        return jsonify({
            "status": "failed",
            "message": f"Download failed. Keeping current model ({MODEL_INFO.get('model')}). Error: {str(e)}"
        }), 500

    if model_path.exists():
        new_model, log_msg = load_model(model_path, json_data)

        if new_model:
            MODEL = new_model
            MODEL_INFO.update(json_data)
            app.logger.info(f"Model successfully changed from {current_model_name} to {model_name}.")

            return jsonify({
                "status": "success",
                "message": f"Model {model_name} downloaded from WandB and loaded successfully."
            })
    app.logger.error(f"Unknown error (file missing after download). Current model set back to {current_model_name}.")
    return jsonify({"status": "failed", "message": "Unknown error: File not found after download."}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests to predict goal probability.
    """
    global MODEL, MODEL_INFO

    if MODEL is None:
        app.logger.error("Prediction attempted without loaded model.")
        return jsonify({"status": "error", "message": "No model loaded. Call /download_registry_model first."}), 503

    try:
        json_data = request.get_json(force=True)
        input_df = pd.read_json(json.dumps(json_data), orient='columns')
    except Exception as e:
        app.logger.error(f"Data parsing failed: {e}")
        return jsonify({"status": "error", "message": "Invalid Data Format"}), 400

    model_name = MODEL_INFO.get("model", "")

    predictions = []

    try:
        if model_name == "RL-Distance":
            if 'distanceToNet' not in input_df.columns:
                raise ValueError("Missing feature: distanceToNet")

            X = input_df[['distanceToNet']]
            predictions = MODEL.predict_proba(X)[:, 1]
        elif model_name == "RL-Angle":
            if 'shotAngle' not in input_df.columns:
                raise ValueError("Missing feature: shotAngle")

            X = input_df[['shotAngle']]
            predictions = MODEL.predict_proba(X)[:, 1]
        elif model_name == "RL-DistanceAngle":
            required = ['distanceToNet', 'shotAngle']
            if not all(col in input_df.columns for col in required):
                raise ValueError(f"Missing features. Required: {required}")

            X = input_df[required]
            predictions = MODEL.predict_proba(X)[:, 1]

        else:
            msg = f"Model type '{model_name}' not supported by feature selector."
            app.logger.error(msg)
            return jsonify({"status": "error", "message": msg}), 400

        response_df = pd.DataFrame({"expected_goal_prob": predictions.round(4)})
        return jsonify({"predictions": response_df.to_json(orient='records')})

    except Exception as e:
        app.logger.error(f"Prediction Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)