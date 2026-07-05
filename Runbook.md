# mlops-gcp-pipeline — Runbook

Pure setup instructions. For the story behind these decisions, see
`Progress.md`.

---

## 1. GCP project setup

```bash
# Create project, then link billing via console.

gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud auth application-default login

gcloud services enable \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  container.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  composer.googleapis.com
```

Region: `europe-central2` (all resources — VM, GCS, BigQuery, GKE, Composer
must use this region for consistency, avoid cross-region latency/egress).

---

## 2. GPU quota check

```bash
gcloud compute regions describe europe-central2 \
  --project=<PROJECT_ID> \
  --format=json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if 'GPU' in q['metric'] or 'T4' in q['metric']:
        print(f\"{q['metric']:30s} limit={q['limit']:<8} usage={q['usage']}\")
"
```
If `NVIDIA_T4_GPUS` limit is 0, request a quota increase before continuing.

---

## 3. Create the training VM

```bash
gcloud compute instances create cifar10-train-vm \
  --project=<PROJECT_ID> \
  --zone=europe-central2-b \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --metadata="install-nvidia-driver=True" \
  --scopes=https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/cloud-platform
```
If `ZONE_RESOURCE_POOL_EXHAUSTED`, try `europe-central2-a` or `-c`, or
check available zones:
```bash
gcloud compute accelerator-types list --filter="name=nvidia-tesla-t4" --format="table(zone)"
```
If the image family above no longer resolves, find the current one:
```bash
gcloud compute images list --project=deeplearning-platform-release \
  --filter="family ~ pytorch.*cu" --format="value(family)" | sort -u
```

**Note:** `--scopes` includes storage read/write from the start — set this
correctly at creation to avoid a later fix requiring a stop/start cycle.

---

## 4. GCS bucket + BigQuery setup

**Run from the local Mac terminal**, not the VM. Bucket/dataset creation
and IAM grants need broader permissions than the VM's service account has
by default — IAM policy changes in particular always require your
personal account, regardless of what scopes the VM was created with.

```bash
gsutil mb -p <PROJECT_ID> -l europe-central2 gs://<PROJECT_ID>-cifar10-artifacts

bq mk --project_id=<PROJECT_ID> --location=europe-central2 mlops_predictions

bq mk --project_id=<PROJECT_ID> \
  --table <PROJECT_ID>:mlops_predictions.predictions \
  timestamp:TIMESTAMP,predicted_class:STRING,confidence:FLOAT,image_filename:STRING,image_gcs_path:STRING,model_version:STRING
```

Grant the VM's service account access (find the account via `gcloud compute
instances describe <vm> --format="value(serviceAccounts[0].email)"`,
run from local):
```bash
gsutil iam ch serviceAccount:<SERVICE_ACCOUNT_EMAIL>:roles/storage.objectAdmin gs://<PROJECT_ID>-cifar10-artifacts
```

**Verify write access — run from the VM:**
```bash
echo "test" | gsutil cp - gs://<PROJECT_ID>-cifar10-artifacts/permission_check.txt
gsutil cat gs://<PROJECT_ID>-cifar10-artifacts/permission_check.txt
gsutil rm gs://<PROJECT_ID>-cifar10-artifacts/permission_check.txt
```

---

## 5. SSH in, install code-server

**SSH connection and code-server install — run from local to connect,
then commands inside the SSH session run on the VM:**
```bash
gcloud compute ssh cifar10-train-vm --project=<PROJECT_ID> --zone=europe-central2-b
nvidia-smi   # confirm GPU visible

curl -fsSL https://code-server.dev/install.sh | sh
code-server --bind-addr 0.0.0.0:8080
cat ~/.config/code-server/config.yaml   # get password
```

**From a separate local terminal tab**, tunnel in:
```bash
gcloud compute ssh cifar10-train-vm --project=<PROJECT_ID> --zone=europe-central2-b -- -L 8080:localhost:8080
```
Browser: `http://localhost:8080`

---

## 6. Clone repo, set up environment

```bash
git clone https://github.com/pg-gitcommits/mlops-gcp-pipeline.git
cd mlops-gcp-pipeline

sudo apt update
sudo apt install -y python3.10-venv
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True Tesla T4
```

Confirm `requirements.txt` includes (add if missing):
```
google-cloud-storage==3.12.0
python-dotenv==1.2.2
```

---

## 7. Create `.env`

```bash
cat > .env << 'EOF'
MODEL_CHECKPOINT=gs://<PROJECT_ID>-cifar10-artifacts/checkpoints/best_model.pt
GCP_PROJECT=<PROJECT_ID>
GCS_BUCKET=<PROJECT_ID>-cifar10-artifacts
BQ_DATASET=mlops_predictions
BQ_TABLE=predictions
MODEL_VERSION=v1
PREDICT_RATE_LIMIT=10/minute
MAX_UPLOAD_BYTES=5242880
EOF
```
Already covered by `.gitignore`'s `*.env` pattern — do not commit.

---

## 8. Install Docker (on the VM)

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
docker --version
```

---

## 9. W&B login

```bash
wandb login
```
Paste API key from https://wandb.ai/authorize

---

## 10. Start MLflow (GCS-backed)

In a `tmux` session (so it survives disconnects):
```bash
tmux new -s mlflow
source venv/bin/activate
mlflow server --host 0.0.0.0 --port 5000 \
  --default-artifact-root gs://<PROJECT_ID>-cifar10-artifacts/mlruns
```
Detach: `Ctrl+b d`

Browser access — tunnel from local terminal:
```bash
gcloud compute ssh cifar10-train-vm --project=<PROJECT_ID> --zone=europe-central2-b -- -L 5000:localhost:5000
```
Then `http://localhost:5000`

---

## 11. Run training

In its own `tmux` session:
```bash
tmux new -s training
source venv/bin/activate
python3 -m src.train --config configs/train_config.yaml
```
Detach: `Ctrl+b d`. Reattach: `tmux attach -t training`.

To force retraining even if this exact hyperparameter combo already ran:
```bash
FORCE_RETRAIN=true python3 -m src.train --config configs/train_config.yaml
```

Check the promoted model:
```bash
gsutil cat gs://<PROJECT_ID>-cifar10-artifacts/checkpoints/best_model.json
```

---

## 12. Run evaluation

```bash
python3 -m src.evaluate --config configs/train_config.yaml
```
Outputs: confusion matrix, classification report, raw predictions CSV in
`logs/`. Evaluates every `.pt` in `checkpoints/` except `best_model.pt`
(alias, not an individual run — skipped automatically).

---

## 12b. Check the currently promoted best model

```bash
gsutil cat gs://<PROJECT_ID>-cifar10-artifacts/checkpoints/best_model.json
```
This is the source of truth for which checkpoint is deployed — no need to
manually compare accuracy numbers, `train.py` already did that at
promotion time.

---

## 13. Build Docker images

```bash
docker build -f docker/Dockerfile.train -t cifar10-train:latest .
docker build -f docker/Dockerfile.api -t cifar10-api:latest .
docker images   # confirm both present
```

---

## 13b. Push images to Artifact Registry

**Repository creation and IAM grant — run from the local Mac terminal.**
The VM's service account lacks Artifact Registry access by default, and
IAM grants always require your personal account regardless of VM scopes:
```bash
gcloud artifacts repositories create cifar10-images \
  --project=<PROJECT_ID> \
  --location=europe-central2 \
  --repository-format=docker

gcloud artifacts repositories add-iam-policy-binding cifar10-images \
  --project=<PROJECT_ID> \
  --location=europe-central2 \
  --member="serviceAccount:<SERVICE_ACCOUNT_EMAIL>" \
  --role="roles/artifactregistry.writer"
```

**Push itself — run from the VM** (where the images live):
```bash
gcloud auth configure-docker europe-central2-docker.pkg.dev

docker tag cifar10-train:latest europe-central2-docker.pkg.dev/<PROJECT_ID>/cifar10-images/cifar10-train:latest
docker tag cifar10-api:latest europe-central2-docker.pkg.dev/<PROJECT_ID>/cifar10-images/cifar10-api:latest

docker push europe-central2-docker.pkg.dev/<PROJECT_ID>/cifar10-images/cifar10-train:latest
docker push europe-central2-docker.pkg.dev/<PROJECT_ID>/cifar10-images/cifar10-api:latest
```

---

## 14. Run the API container

```bash
docker run --env-file .env -p 8000:8000 cifar10-api:latest
```

Test (from inside the VM):
```bash
curl http://localhost:8000/health
curl http://localhost:8000/classes
curl -X POST http://localhost:8000/predict -F "file=@/path/to/image.jpg"
```

Browser access (Swagger UI) — tunnel from local terminal:
```bash
gcloud compute ssh cifar10-train-vm --project=<PROJECT_ID> --zone=europe-central2-b -- -L 8000:localhost:8000
```
Then `http://localhost:8000/docs`

---

## 15. Run tests

```bash
python3 -m pytest tests/ -v
```
Requires `httpx`, `pytest`, `slowapi` in `requirements.txt` — confirm
present before running:
```bash
grep -E "^httpx==|^pytest==|^slowapi==" requirements.txt
```

Tests must not depend on ambient `.env` state — anything reading
`MODEL_CHECKPOINT` or similar globals from `api/main.py` needs an
explicit `patch('api.main.<VAR>', ...)`, since `.env` is gitignored and
won't exist in CI.

---

## 16. CI/CD one-time setup

**Service account for GitHub Actions — run from local:**
```bash
gcloud iam service-accounts create github-actions-ci \
  --project=<PROJECT_ID> \
  --display-name="GitHub Actions CI/CD"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:github-actions-ci@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:github-actions-ci@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/container.developer"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:github-actions-ci@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=github-actions-ci@<PROJECT_ID>.iam.gserviceaccount.com
```

If key creation fails with `constraints/iam.disableServiceAccountKeyCreation`,
this is an org-level policy, not a project permission issue. Fix (run
from local, needs org-level rights):
```bash
gcloud organizations list
# get ORG_ID

gcloud organizations add-iam-policy-binding ORG_ID \
  --member="user:<YOUR_EMAIL>" \
  --role="roles/orgpolicy.policyAdmin"

gcloud resource-manager org-policies disable-enforce \
  constraints/iam.disableServiceAccountKeyCreation \
  --project=<PROJECT_ID>
```
Then retry key creation.

**GitHub repo secrets** (Settings → Secrets and variables → Actions):
- `GCP_SA_KEY` → full contents of `github-actions-key.json`
- `GCP_PROJECT_ID` → `<PROJECT_ID>`
- `GCP_REGION` → `europe-central2`
- `COMPOSER_BUCKET` → leave unset until Composer exists (Section 19).
  Once set, every future push to `main` auto-uploads DAGs to the live
  Airflow environment — a deliberate step, not automatic.

**Delete the local key file after pasting into GitHub:**
```bash
rm github-actions-key.json
```

---

## 17. Commit and push (triggers CI/CD)

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

git status   # review before staging — check for stray files (installer
             # scripts, test images) and confirm mlflow.db/logs/ tracking
             # matches intent

git add .
git commit -m "Describe the change"
git push origin main
```

Check the Actions tab on GitHub to watch the pipeline run.
`build-and-push` and `deploy` only run if `test` passes first — a broken
commit never reaches Artifact Registry or GKE.

**Note:** `deploy` and `deploy-dags` showing green does not mean anything
was deployed — both jobs guard-check whether GKE/Composer exist first and
exit successfully either way (deploy or skip). Check the actual step log
text to see which branch ran, not the checkmark color.

---

## 18. GKE deployment (Autopilot)

**k8s manifests needed:** `k8s/configmap.yaml`, `k8s/deployment.yaml`,
`k8s/service.yaml`. ConfigMap holds all API env vars; deployment
references it via `envFrom`. Confirm `image:` in `deployment.yaml` points
at the correct region/project/repo before applying.

**Create the cluster — run from local:**
```bash
gcloud container clusters create-auto cifar10-cluster \
  --project=<PROJECT_ID> \
  --region=europe-central2
```

**Grant kubectl access from wherever you run it** (VM shown here — if
running from local Mac instead, skip straight to `get-credentials`, your
personal account already has full access):
```bash
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:<VM_SERVICE_ACCOUNT_EMAIL>" \
  --role="roles/container.developer"
```
VM also needs `cloud-platform` AND `userinfo.email` scopes (the latter is
not included in `cloud-platform` despite sounding like it should be) —
set both at VM creation time (see Section 3) to avoid a stop/start cycle
later.

**On the VM, install kubectl tooling if not already present:**
```bash
sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin
sudo snap install kubectl --classic
```

**Connect kubectl to the cluster:**
```bash
gcloud container clusters get-credentials cifar10-cluster \
  --project=<PROJECT_ID> \
  --region=europe-central2
```

**Workload Identity setup — GKE Autopilot always requires this, cannot
be skipped.** Pods get zero GCP permissions by default; each workload
needs its own explicit identity. Run from local:
```bash
gcloud iam service-accounts create cifar10-api-sa \
  --project=<PROJECT_ID> \
  --display-name="CIFAR10 API Pod"

gsutil iam ch serviceAccount:cifar10-api-sa@<PROJECT_ID>.iam.gserviceaccount.com:roles/storage.objectAdmin gs://<PROJECT_ID>-cifar10-artifacts

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:cifar10-api-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

gcloud iam service-accounts add-iam-policy-binding \
  cifar10-api-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:<PROJECT_ID>.svc.id.goog[default/cifar10-api-ksa]"
```

**Create and link the Kubernetes-side ServiceAccount — run on the VM:**
```bash
kubectl create serviceaccount cifar10-api-ksa
kubectl annotate serviceaccount cifar10-api-ksa iam.gke.io/gcp-service-account=cifar10-api-sa@<PROJECT_ID>.iam.gserviceaccount.com
```

Confirm `deployment.yaml` includes, under `spec:` (before `containers:`):
```yaml
serviceAccountName: cifar10-api-ksa
```

**Apply everything:**
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

**Verify:**
```bash
kubectl get pods            # RESTARTS should stay at 0
kubectl logs -l app=cifar10-api
kubectl get service cifar10-api-service   # get EXTERNAL-IP
curl http://<EXTERNAL-IP>/health
```

`kubectl get nodes` returning empty before anything is applied is normal
for Autopilot — nodes provision on demand, not upfront.

**Delete the cluster when not actively using it** (Autopilot has no
"stop," only delete/recreate — no working pod means no reason to pay for
it overnight):
```bash
gcloud container clusters delete cifar10-cluster --project=<PROJECT_ID> --region=europe-central2
```

---

## 19. Airflow (Cloud Composer)

**DAG design note:** training and evaluation run via SSH to the
always-on GPU VM, not as GKE pods — a deliberate choice tied to a T4
quota of 1. See the production-scale comment at the bottom of
`dags/training_pipeline.py` for what changes once quota allows a proper
move to GKE with a GPU node pool.

**Cost warning:** Composer has no pause/stop — only create/delete. A
small environment costs roughly $12-13/day while it exists, running or
idle. Delete it when not actively testing.

**Create a dedicated service account first** — Composer requires one be
explicitly specified:
```bash
gcloud iam service-accounts create cifar10-composer-sa \
  --project=<PROJECT_ID> \
  --display-name="Cloud Composer Environment"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:cifar10-composer-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/composer.worker"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:cifar10-composer-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:cifar10-composer-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"
```
(`bigquery.jobUser` is required in addition to `dataViewer` — the latter
only allows reading table contents, not running a query job.)

**Create the environment — run from local.** Check current valid image
versions if this errors (`gcloud composer environments create ... `
without `--image-version` won't tell you; the error message from an
invalid version lists the current valid ones):
```bash
gcloud composer environments create cifar10-composer \
  --project=<PROJECT_ID> \
  --location=europe-central2 \
  --image-version=composer-3-airflow-2.11.1 \
  --service-account=cifar10-composer-sa@<PROJECT_ID>.iam.gserviceaccount.com
```
Takes 20-30 minutes.

**Grant the VM access to Composer's auto-created bucket** (separate from
the project's main artifacts bucket — find its name via the describe
command below, then grant access — run from local):
```bash
gcloud composer environments describe cifar10-composer \
  --project=<PROJECT_ID> --location=europe-central2 \
  --format="value(config.dagGcsPrefix)"

gsutil iam ch serviceAccount:<VM_SERVICE_ACCOUNT_EMAIL>:roles/storage.objectAdmin gs://<COMPOSER_BUCKET_NAME>
```

**Upload DAGs — run on the VM:**
```bash
gsutil cp dags/*.py gs://<COMPOSER_BUCKET_NAME>/dags/
```

**Set up the SSH connection Airflow needs to reach the VM.** Generate a
dedicated key (not your personal one), register the public half on the
VM, upload the private half to Composer's bucket:
```bash
ssh-keygen -t rsa -f ~/airflow-vm-key -C airflow-composer -N ""

gcloud compute instances add-metadata cifar10-train-vm \
  --project=<PROJECT_ID> --zone=europe-central2-b \
  --metadata-from-file ssh-keys=<(echo "<VM_USERNAME>:$(cat ~/airflow-vm-key.pub)")

gsutil cp ~/airflow-vm-key gs://<COMPOSER_BUCKET_NAME>/data/airflow-vm-key
```

Then, in the **Airflow UI** (get the URL via
`gcloud composer environments describe cifar10-composer --project=<PROJECT_ID> --location=europe-central2 --format="value(config.airflowUri)"`),
go to **Admin → Connections → +** and add manually (the CLI equivalent,
`gcloud composer environments run ... connections add`, has a known
crash bug — use the UI instead):
- Connection Id: `cifar10_vm_ssh`
- Connection Type: `SSH`
- Host: VM's external IP
- Login: VM username
- Extra: `{"key_file": "/home/airflow/gcs/data/airflow-vm-key"}`

**Before triggering training or evaluation, MLflow must already be
running on the VM** — the pipelines check for this and fail fast if not
(don't skip starting it):
```bash
tmux new -s mlflow
cd ~/mlops-gcp-pipeline && source venv/bin/activate
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root gs://<PROJECT_ID>-cifar10-artifacts/mlruns
```

**Trigger a DAG manually** (bypasses schedule entirely) via the UI's ▶️
button, or:
```bash
airflow dags trigger training_pipeline
```

**Critical operational rule:** marking a task failed or retrying it in
the Airflow UI does **not** stop the actual remote process on the VM —
UI state and real VM processes are disconnected once an SSH command has
been dispatched. If a run needs to be stopped, kill the real PID
directly:
```bash
ssh into the VM
ps aux | grep train
kill <pid>
```
Both `training_pipeline.py` and `evaluation_pipeline.py` have
`max_active_runs=1` set specifically to prevent multiple concurrent runs
from stacking on the single GPU — don't remove this.

**Delete the Composer environment when done testing:**
```bash
gcloud composer environments delete cifar10-composer --project=<PROJECT_ID> --location=europe-central2
```

---

## 20. Drift monitoring

**Setup — on the VM:**
```bash
mkdir -p monitoring
touch monitoring/__init__.py
```
Save `drift_detector.py` to `monitoring/drift_detector.py`. Confirm
`google-cloud-bigquery` is in `requirements.txt` (training venv) —
`requirements-api.txt` alone isn't enough:
```bash
grep "^google-cloud-bigquery==" requirements.txt || echo "google-cloud-bigquery==<version>" >> requirements.txt
```

**Run a manual check:**
```bash
python3 -m monitoring.drift_detector
```
Prints results as JSON and writes a timestamped copy to
`monitoring/results/`.

**Note on sparse data:** with very few real predictions logged, most of
the 10 classes will have zero count in both the baseline and recent
windows. `drift_detector.py` already handles this — skips zero-expected
classes from the chi-squared test, returns `reason:
"insufficient_class_coverage"` with `p_value: None` if fewer than 2
classes have real data, rather than a silent `NaN`.

**Upload to Composer** (needs both the DAG and the module, correctly
nested — from the VM):
```bash
gsutil cp -r monitoring gs://<COMPOSER_BUCKET_NAME>/dags/
gsutil cp dags/monitoring_dag.py gs://<COMPOSER_BUCKET_NAME>/dags/
```

---

## Every-session start/stop

**Stop VM (end of day, avoids billing):**
```bash
gcloud compute instances stop cifar10-train-vm --project=<PROJECT_ID> --zone=europe-central2-b
```

**Start VM + reconnect:**
```bash
gcloud compute instances start cifar10-train-vm --project=<PROJECT_ID> --zone=europe-central2-b
gcloud compute ssh cifar10-train-vm --project=<PROJECT_ID> --zone=europe-central2-b
```

code-server, MLflow, and tmux sessions do NOT survive a stop/start — must
be relaunched each session (repeat relevant commands from sections 5, 10).

**All tunnel commands run from your local machine, never from inside the VM.**
