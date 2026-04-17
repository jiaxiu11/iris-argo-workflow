import os
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

run_id = os.environ["MLFLOW_RUN_ID"]
accuracy = os.environ["ACCURACY"]

result = mlflow.register_model(f"runs:/{run_id}/model", "iris-classifier")

client = MlflowClient()
client.set_model_version_tag("iris-classifier", result.version, "accuracy", accuracy)
print(f"Registered iris-classifier v{result.version} (accuracy={accuracy})")
