import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("iris-pipeline")

train_df = pd.read_csv('/mnt/data/train.csv')
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

params = {"n_estimators": 100, "random_state": 42, "max_depth": 5}
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)

with mlflow.start_run() as run:
    mlflow.log_params(params)
    mlflow.log_metric("train_accuracy", train_score)
    mlflow.sklearn.log_model(model, "model")
    run_id = run.info.run_id

with open('/tmp/run_id.txt', 'w') as f:
    f.write(run_id)
print(f"Model logged to MLflow run {run_id}")
