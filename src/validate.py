import os
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
run_id = os.environ["MLFLOW_RUN_ID"]

test_df = pd.read_csv('/mnt/data/test.csv')
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Validation accuracy: {accuracy:.4f}")

with mlflow.start_run(run_id=run_id):
    mlflow.log_metric("val_accuracy", accuracy)

with open('/tmp/accuracy.txt', 'w') as f:
    f.write(str(round(accuracy, 4)))
