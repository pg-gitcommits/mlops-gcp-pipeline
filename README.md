# mlops-gcp-pipeline

A production-grade MLOps pipeline built on Google Cloud Platform, demonstrating the full ML lifecycle from model training to deployment and monitoring. Built using CIFAR-10 image classification as the ML problem, with ResNet-18 trained in PyTorch.

---

## Architecture Overview

```
GitHub Repository
      ↓ push
GitHub Actions (CI/CD)
      ↓ build + test + push
GCP Artifact Registry
      ↓ deploy
GKE Inference API (FastAPI + ResNet-18)
      ↓ predictions logged
BigQuery Prediction Logs + GCS Image Storage
      ↓ daily drift detection
Cloud Composer Airflow
      ↓ drift detected
Automated Retraining Pipeline (SSH-triggered on GPU VM)
      ↓ new model
GKE Inference API (updated)
```

**Note on training location:** scheduled/triggered training runs via
Airflow SSH into a dedicated, always-on GPU VM rather than running as a
GKE pod. This is a deliberate choice tied to a T4 GPU quota of 1 — see
`dags/training_pipeline.py`'s docstring and the comment at the bottom of
that file for what changes once quota allows a move to GKE with a GPU
node pool (the more conventional production pattern for ephemeral,
autoscaling training compute).

---

## Tech Stack

| Component           | Technology                             |
| ------------------- | --------------------------------------- |
| ML Framework        | PyTorch                                |
| Model               | ResNet-18 (pretrained, fine-tuned)     |
| Experiment Tracking | MLflow + Weights & Biases              |
| Containerisation    | Docker                                 |
| Container Registry  | GCP Artifact Registry                  |
| CI/CD               | GitHub Actions                         |
| Model Serving       | FastAPI                                |
| Deployment          | Google Kubernetes Engine (GKE Autopilot) |
| Prediction Logging  | BigQuery + GCS                         |
| Orchestration       | Cloud Composer (Airflow)               |
| Cloud Platform      | Google Cloud Platform                  |

---

## Project Structure

```
mlops-gcp-pipeline/
├── src/
│   ├── model.py              # ResNet-18 architecture
│   ├── train.py              # Training script with grid search + GCS checkpoint promotion
│   ├── evaluate.py           # Evaluation with confusion matrix
│   └── dataset.py            # CIFAR-10 data loading
├── api/
│   ├── main.py               # FastAPI inference server
│   ├── models.py             # Request/response schemas
│   └── utils.py              # Image preprocessing
├── docker/
│   ├── Dockerfile.train      # Training container (CUDA)
│   └── Dockerfile.api        # Inference container (CPU)
├── k8s/
│   ├── configmap.yaml        # API environment configuration
│   ├── deployment.yaml       # Kubernetes deployment
│   └── service.yaml          # LoadBalancer service
├── dags/
│   ├── training_pipeline.py  # Airflow training DAG (SSH to GPU VM)
│   ├── evaluation_pipeline.py
│   ├── retraining_trigger.py
│   └── monitoring_dag.py     # Drift detection DAG
├── monitoring/
│   └── drift_detector.py     # Standalone drift detection (CLI + importable by the DAG)
├── tests/
│   ├── test_model.py         # Model unit tests
│   └── test_api.py           # API unit tests
├── configs/
│   └── train_config.yaml     # Hyperparameters and paths
└── .github/
    └── workflows/
        └── ci.yml            # CI/CD pipeline
```

---

## Phase 1 — Model Training

- ResNet-18 fine-tuned on CIFAR-10 (10 classes, 60,000 images)
- Transfer learning from ImageNet weights
- Grid search across learning rates: `[0.01, 0.001]`
- SGD optimiser with momentum=0.9, weight_decay=0.0001
- StepLR scheduler (step_size=7, gamma=0.1)
- Training on GCP VM with NVIDIA T4 GPU
- Checkpoints uploaded to GCS after every run; the best-performing
  checkpoint is promoted to a stable `best_model.pt` alias (with a JSON
  sidecar recording which run it came from and a rollback history log),
  so downstream consumers (the API, evaluation) never need to track a
  specific timestamped filename

### Results

| Run | Architecture | LR    | Epochs | Best Val Acc |
| --- | ------------ | ----- | ------ | ------------ |
| 1   | ResNet-18    | 0.01  | 20     | **83.45%**   |
| 2   | ResNet-18    | 0.001 | 20     | 80.39%       |

**Confusion matrix — best model (ResNet-18, lr=0.01, 83.45% val acc):**

![Confusion matrix](logs/confusion_matrix_resnet18_ep020_vacc83.45_20260629.png)

Full classification report:
[`logs/classification_report_resnet18_ep020_vacc83.45_20260629.txt`](logs/classification_report_resnet18_ep020_vacc83.45_20260629.txt)

Known open issue: cat/dog classification precision is noticeably lower
than other classes — visible in the confusion matrix above — not yet
investigated as of this restart.

---

## Phase 2 — Experiment Tracking

Two experiment tracking tools used simultaneously.

### MLflow

- Self-hosted on the GPU VM, GCS-backed artifact storage
  (`--default-artifact-root gs://<bucket>/mlruns`)
- Logs parameters, metrics, and artifacts per run
- Tags: `best_val_acc`, `dataset`, `checkpoint`, `date`

### Weights & Biases

- Cloud-hosted SaaS
- Real-time training curves during active training
- Hyperparameter comparison across runs
- GPU utilisation monitoring

### MLflow vs W&B

|                  | MLflow               | W&B                    |
| ---------------- | --------------------- | ----------------------- |
| Hosting          | Self-hosted           | Cloud SaaS              |
| Real-time charts | No                    | Yes                     |
| Cost at scale    | Free                  | Paid                    |
| Best for         | Production pipelines  | Active experimentation  |

---

## Phase 3 — Containerisation

Two Docker images built and pushed to GCP Artifact Registry:

**Training image** (`cifar10-train`) — CUDA-enabled PyTorch base image, packages `src/` and `configs/`. Used for GPU-accelerated training.

**Inference image** (`cifar10-api`) — CPU-only PyTorch, packages `api/` and `src/`. Minimal dependencies for fast startup and low cost.

Both images tagged with `latest` and the Git commit SHA for full traceability and rollback capability.

---

## Phase 4 — CI/CD with GitHub Actions

On every push to `main` (for relevant file changes):

1. **Run Tests** — pytest runs all unit tests for model and API
2. **Build Docker Images** — both training and inference images built
3. **Push to Artifact Registry** — images tagged with `latest` and commit SHA
4. **Deploy to GKE** — inference API updated if the cluster exists
5. **Deploy DAGs** — Airflow DAGs (and the `monitoring/` package) synced to the Cloud Composer bucket if Composer is configured

Both deploy steps degrade gracefully — they check whether their target
infrastructure exists first and skip cleanly if not, rather than failing
the whole pipeline. A green checkmark on either step means it completed
successfully, which includes a deliberate skip — check the step's log
text, not just the checkmark, to see which branch actually ran.

Path filtering ensures CI only triggers when relevant files change: `src/`, `api/`, `docker/`, `tests/`, `configs/`, `dags/`, `k8s/`, `monitoring/`

---

## Phase 5 — Kubernetes Deployment

Inference API deployed to a GKE **Autopilot** cluster (chosen over Standard mode — billed per pod resource request rather than per provisioned node, which fits a single small API service without paying for idle node capacity):

- **Deployment:** 1 replica, environment configured via a ConfigMap (`k8s/configmap.yaml`)
- **Service:** LoadBalancer with public IP
- **Health checks:** Liveness and readiness probes on `/health`
- **Model loading:** Checkpoint downloaded from GCS at startup (`gs://<bucket>/checkpoints/best_model.pt`)
- **Prediction logging:** Every prediction logged to BigQuery + image saved to GCS
- **Identity:** the pod runs under its own dedicated Google service account via Workload Identity Federation (mandatory and always-on in GKE Autopilot — pods never inherit the node's default permissions)

### API Endpoints

| Endpoint   | Method | Description                  |
| ---------- | ------ | ----------------------------- |
| `/health`  | GET    | Health check                  |
| `/predict` | POST   | Upload image, get prediction  |
| `/classes` | GET    | List all 10 classes           |
| `/docs`    | GET    | Interactive Swagger UI        |

### Prediction Logging Schema (BigQuery)

| Column            | Type      | Description                 |
| ------------------ | --------- | ---------------------------- |
| `timestamp`        | TIMESTAMP | When prediction was made     |
| `predicted_class`  | STRING    | Predicted CIFAR-10 class     |
| `confidence`       | FLOAT     | Model confidence (%)         |
| `image_filename`   | STRING    | UUID filename                 |
| `image_gcs_path`   | STRING    | GCS path to uploaded image   |
| `model_version`    | STRING    | Model version identifier     |

---

## Phase 6 — Airflow Orchestration

Four DAGs deployed to Cloud Composer:

- `training_pipeline` — SSH-triggers training on the dedicated GPU VM (not a GKE pod — see the architecture note above), sets `FORCE_RETRAIN=true` so scheduled/triggered runs always retrain regardless of matching hyperparameters, with pre-flight checks that the VM/GPU and MLflow are actually reachable before starting
- `evaluation_pipeline` — same SSH pattern, runs evaluation and uploads reports to GCS
- `retraining_trigger` — triggers training then evaluation sequentially
- `monitoring_dag` — daily drift detection (`@daily`), triggers retraining automatically if drift is detected

Both `training_pipeline` and `evaluation_pipeline` set `max_active_runs=1`
to prevent concurrent runs from stacking on the single shared GPU.

---

## Phase 7 — Monitoring and Drift Detection

- Prediction logs collected in BigQuery (Phase 5)
- Drift detection compares a recent (last 24h) predicted-class
  distribution against a baseline (first week of predictions) using a
  Chi-squared test and Jensen-Shannon Divergence
- Detection logic lives in `monitoring/drift_detector.py`, shared by both
  `monitoring_dag.py` and a standalone CLI:
  ```bash
  python3 -m monitoring.drift_detector
  ```
  Prints results as JSON and writes a timestamped copy to
  `monitoring/results/`.
- Handles sparse-data edge cases explicitly: classes with zero
  representation in both comparison windows are excluded from the
  chi-squared test (avoiding a `0/0`-driven `NaN` result); if fewer than
  2 classes have real data, returns a clear `insufficient_class_coverage`
  reason instead of a misleading statistic.

---

## How to Run

See `Runbook.md` for the complete, step-by-step setup sequence (GCP
project setup through drift monitoring). Summary below.

### Prerequisites

- GCP project with billing enabled
- `gcloud` CLI authenticated
- Docker installed
- Python 3.10+

### 1. Clone the repository

```bash
git clone https://github.com/pg-gitcommits/mlops-gcp-pipeline.git
cd mlops-gcp-pipeline
```

### 2. Set up environment

Training environment needs the CUDA-matched torch build (see
`requirements.txt`'s pinned `+cu121` versions):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### 3. Configure

Create a `.env` file at the repo root (gitignored):
```
MODEL_CHECKPOINT=gs://<your-bucket>/checkpoints/best_model.pt
GCP_PROJECT=<your-project-id>
GCS_BUCKET=<your-bucket>
BQ_DATASET=mlops_predictions
BQ_TABLE=predictions
MODEL_VERSION=v1
PREDICT_RATE_LIMIT=10/minute
MAX_UPLOAD_BYTES=5242880
```
Both `train.py` and `api/main.py` load this automatically via
`python-dotenv`.

### 4. Train

```bash
python3 -m src.train --config configs/train_config.yaml
```
Set `FORCE_RETRAIN=true` to bypass the skip-if-already-tried check (used
by Airflow for scheduled retraining).

### 5. Evaluate

```bash
python3 -m src.evaluate --config configs/train_config.yaml
```

### 6. Run tests

```bash
python3 -m pytest tests/ -v
```

### 7. Build and run inference API locally

```bash
docker build -f docker/Dockerfile.api -t cifar10-api:latest .
docker run --env-file .env -p 8000:8000 cifar10-api:latest
```
Open `http://localhost:8000/docs` to test predictions.

### 8. Deploy to GKE

```bash
gcloud container clusters create-auto cifar10-cluster \
  --project=<your-project-id> \
  --region=<your-region>

kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl get service cifar10-api-service
```
Requires Workload Identity setup for the pod's GCS/BigQuery access — see
`Runbook.md` section 18.

---

## GitHub Actions Secrets Required

| Secret            | Description                                         |
| ------------------ | ----------------------------------------------------- |
| `GCP_SA_KEY`       | GCP service account JSON key                          |
| `GCP_PROJECT_ID`   | GCP project ID                                        |
| `GCP_REGION`       | GCP region                                            |
| `COMPOSER_BUCKET`  | Cloud Composer GCS bucket name (unset until Composer exists) |

---

## Known Open Items

- **Cat/dog classification precision** is noticeably lower than other
  classes — not yet investigated.
- **Training runs on a dedicated GPU VM via SSH, not GKE** — a deliberate
  choice given a T4 GPU quota of 1. See the note at the bottom of
  `dags/training_pipeline.py` for the production-scale migration path
  (GKE GPU node pool + MLflow moved off the VM onto a stable, independent
  address).
- **No API authentication** — `/predict` is currently public on the
  LoadBalancer IP.
- **IAM roles are broader than strictly necessary** in places (e.g.
  `storage.objectAdmin` includes delete, when read/write would suffice).
- **Infrastructure is not managed as code** — every GCP resource was
  created via `gcloud`/`gsutil`/`bq` and documented in `Runbook.md`, not
  Terraform.
- **`train.py` doesn't clean up orphaned `temp_` checkpoint files** left
  behind by a non-clean training interruption (dropped SSH, killed
  process, VM restart) — `evaluate.py` skips these safely, but they
  accumulate on disk.

---

## License

MIT