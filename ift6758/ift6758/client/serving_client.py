import json
import requests
import pandas as pd
import logging
from io import StringIO


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        url = f"{self.base_url}/predict"
        logger.info(f"Sending prediction request to {url} for {len(X)} samples.")
        try:
            payload = X.to_dict(orient='dict')
        except Exception as e:
            logger.error(f"Failed to serialize dataframe: {e}")
            raise e

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return pd.DataFrame()

        try:
            data = response.json()
            preds_df = pd.read_json(StringIO(data["predictions"]), orient='records')
            return preds_df
        except Exception as e:
            logger.error(f"Failed to decode server response: {e}")
            return pd.DataFrame()


    def logs(self) -> dict:
        """Get server logs"""

        url = f"{self.base_url}/logs"
        logger.info(f"Fetching logs from {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get logs: {e}")
            return {"error": str(e)}

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        url = f"{self.base_url}/download_registry_model"
        logger.info(f"Requesting model download: {model} version {version} from {workspace}")

        payload = {
            "workspace": workspace,
            "model": model,
            "version": version
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger model download: {e}")
            return {"status": "failed", "error": str(e)}