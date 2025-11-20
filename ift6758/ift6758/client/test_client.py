import pandas as pd
from serving_client import ServingClient

if __name__ == '__main__':

    client = ServingClient(ip="127.0.0.1", port=5000)

    print("--- Testing Model Download ---")
    response = client.download_registry_model(workspace="local", model="clf_dist_angle", version="1.0.0")
    print(response)

    print("\n--- Testing Prediction ---")
    data = {
        "distanceToNet": [20.5, 50.0],
        "shotAngle": [15.0, 0.0]
    }
    df = pd.DataFrame(data)

    predictions = client.predict(df)
    print("Predictions:")
    print(predictions)

    print("\n--- Testing Logs ---")
    logs = client.logs()
    print(logs['logs'][-3:])