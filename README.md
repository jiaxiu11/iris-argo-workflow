# Iris ML Pipeline

An end-to-end ML pipeline on Kubernetes using Argo Workflows, MLflow, MinIO, and KServe.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           Kubernetes (minikube)          │
                        │                  argo ns                 │
                        │                                          │
  ┌──────────┐          │  ┌────────────────────────────────────┐  │
  │  Argo    │  submit  │  │         Argo Workflow DAG          │  │
  │  CLI /   │─────────▶│  │                                    │  │
  │  kubectl │          │  │  download-data → preprocess        │  │
  └──────────┘          │  │       → train → validate           │  │
                        │  │       → register (if acc > 0.9)    │  │
                        │  └────────────┬───────────────────────┘  │
                        │               │                           │
                        │        params/metrics/model               │
                        │               │                           │
                        │  ┌────────────▼────────────┐             │
                        │  │     MLflow Server        │             │
                        │  │  (tracking + registry)   │             │
                        │  └────────────┬─────────────┘            │
                        │               │ metadata                  │
                        │  ┌────────────▼─────────────┐            │
                        │  │        Postgres           │            │
                        │  │   (MLflow backend store)  │            │
                        │  └──────────────────────────┘            │
                        │                                           │
                        │  ┌───────────────────────────┐           │
                        │  │          MinIO             │           │
                        │  │   (MLflow artifact store)  │           │
                        │  │  s3://mlflow-artifacts/    │           │
                        │  └────────────▲──────────────┘           │
                        │               │                           │
                        │    model artifacts uploaded               │
                        │    directly by MLflow client              │
                        │    in workflow pods                       │
                        │                                           │
                        │  ┌───────────────────────────┐           │
                        │  │   KServe InferenceService  │           │
                        │  │   (kserve-sklearnserver)   │           │
                        │  │                            │           │
                        │  │  storage-initializer pulls │           │
                        │  │  model.pkl from MinIO      │           │
                        │  │  at pod startup            │           │
                        │  └───────────────────────────┘           │
                        └─────────────────────────────────────────┘
```

## Pipeline Steps

| Step | Script | What it does |
|---|---|---|
| `download-data` | `src/download_data.py` | Loads the Iris dataset and writes it to a shared PVC |
| `preprocess` | `src/preprocess.py` | Drops nulls, splits into train/test CSVs |
| `train` | `src/train.py` | Trains a RandomForestClassifier; logs params + `train_accuracy` to MLflow; uploads model artifact to MinIO; writes `run_id` as an Argo output parameter |
| `validate` | `src/validate.py` | Loads model from MinIO via `run_id`; logs `val_accuracy` back to the same MLflow run; writes accuracy for Argo conditional |
| `register` | `src/register.py` | Registers model in MLflow model registry as `iris-classifier` if accuracy > 0.9 |

## MLflow Integration

- **Tracking server** — stores run metadata (params, metrics) in Postgres
- **Artifact store** — model files are uploaded directly from workflow pods to MinIO using boto3 (the MLflow client bypasses the tracking server for artifacts)
- **Model registry** — `register.py` calls `mlflow.register_model()` to create versioned entries in the registry

The MLflow server needs MinIO credentials to serve artifact download URLs in the UI. Workflow pods also carry the same credentials to upload artifacts directly.

## KServe Serving

The `InferenceService` uses the `kserve-sklearnserver` runtime:

1. At pod startup, the **storage initializer** init container authenticates with MinIO and downloads `model.pkl` from the MLflow artifact path to `/mnt/models/`
2. The **sklearn server** loads the pickle and serves predictions

**V2 inference request:**
```bash
curl -X POST http://localhost:8080/v2/models/iris-classifier/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "predict",
      "shape": [1, 4],
      "datatype": "FP64",
      "data": [[5.1, 3.5, 1.4, 0.2]]
    }]
  }'
```

**Response:** `{"outputs": [{"name": "output-0", "datatype": "INT64", "data": [0]}]}`  
Classes: `0` = setosa, `1` = versicolor, `2` = virginica

## Infrastructure (`k8s/`)

| File | What it deploys |
|---|---|
| `secrets.yaml` | MinIO + Postgres credentials |
| `postgres.yaml` | Postgres PVC + Deployment + Service (MLflow backend) |
| `minio.yaml` | MinIO PVC + Deployment + Service + bucket init Job |
| `mlflow.yaml` | MLflow tracking server Deployment + Service |
| `kserve-minio-sa.yaml` | KServe ServiceAccount + annotated Secret for MinIO access |
| `inferenceservice.yaml` | KServe InferenceService pointing at the registered model artifact |

## Prerequisites

- [minikube](https://minikube.sigs.k8s.io/)
- [Argo Workflows](https://argoproj.github.io/argo-workflows/) installed in the `argo` namespace
- [cert-manager](https://cert-manager.io/) installed
- [KServe](https://kserve.github.io/) installed in RawDeployment mode

## Running the Pipeline

```bash
# Deploy infra (first time only)
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/minio.yaml
kubectl apply -f k8s/mlflow.yaml

# Apply the WorkflowTemplate
kubectl apply -f workflow-template.yaml

# Submit a training run
kubectl create -f workflow.yaml

# Watch progress
kubectl get pods -n argo -w --selector='workflows.argoproj.io/workflow=<name>'

# Deploy the InferenceService (after a successful run)
kubectl apply -f k8s/kserve-minio-sa.yaml
kubectl apply -f k8s/inferenceservice.yaml
```

## Accessing UIs

```bash
# MLflow
kubectl port-forward -n argo svc/mlflow 5000:5000
# open http://localhost:5000

# MinIO console
kubectl port-forward -n argo svc/minio 9001:9001
# open http://localhost:9001  (minioadmin / minioadmin123)

# KServe predictor (replace pod name)
kubectl port-forward -n argo pod/<iris-classifier-predictor-xxxxx> 8080:8080
```
