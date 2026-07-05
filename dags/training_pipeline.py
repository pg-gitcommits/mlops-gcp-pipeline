"""
Training Pipeline DAG
Orchestrates CIFAR-10 model training on the dedicated GPU VM
(cifar10-train-vm), triggered manually or by the retraining trigger DAG.

Architecture note: training runs via SSH to a persistent, always-on GPU
VM rather than as a GKE pod. This is a deliberate choice given a T4 GPU
quota of 1 — the VM already holds that quota, and GKE Autopilot has no
GPU node pool provisioned. This is NOT the recommended production
pattern; see the note at the bottom of this file for what changes once
GPU quota allows a proper move to GKE.

Requires an Airflow SSH Connection named 'cifar10_vm_ssh' configured with
the VM's IP/hostname and an SSH key with access to it:
    airflow connections add 'cifar10_vm_ssh' \\
        --conn-type 'ssh' \\
        --conn-host '<VM_EXTERNAL_IP>' \\
        --conn-login 'gowthambulusu' \\
        --conn-extra '{"key_file": "/path/to/private_key"}'
"""

from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'mlops',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

SSH_CONN_ID = 'cifar10_vm_ssh'
REPO_DIR = '~/mlops-gcp-pipeline'

with DAG(
    dag_id='training_pipeline',
    description='Train ResNet-18 on CIFAR-10 via SSH to the GPU VM',
    default_args=default_args,
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['training', 'cifar10', 'resnet18']
) as dag:

    check_vm_and_gpu = SSHOperator(
        task_id='check_vm_and_gpu',
        ssh_conn_id=SSH_CONN_ID,
        command='nvidia-smi',
        cmd_timeout=30,
    )

    check_mlflow_running = SSHOperator(
        task_id='check_mlflow_running',
        ssh_conn_id=SSH_CONN_ID,
        command='curl -sf http://localhost:5000 -o /dev/null',
        cmd_timeout=15,
    )

    train_task = SSHOperator(
        task_id='train_model',
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f'bash -c "cd {REPO_DIR} && '
            'source venv/bin/activate && '
            'export FORCE_RETRAIN=true && '
            'python3 -m src.train --config configs/train_config.yaml"'
        ),
        cmd_timeout=3600,
    )

    check_vm_and_gpu >> check_mlflow_running >> train_task


# --- Production note (GPU quota >= 2) ---
# Move training to GKE with a GPU-enabled node pool:
#   - Autopilot: request the appropriate GPU accelerator type per pod
#     (resources.limits: nvidia.com/gpu), or a Standard cluster with a
#     dedicated GPU node pool that autoscales to zero when idle.
#   - Replace SSHOperator tasks above with GKEStartPodOperator, targeting
#     cifar10-cluster, using the cifar10-train image (already built and
#     pushed to Artifact Registry — no changes needed there).
#   - MLFLOW_TRACKING_URI must change from "http://localhost:5000" to a
#     stable, independently-reachable address — MLflow can no longer live
#     on "whichever VM happens to be running," since training would be
#     ephemeral pods with no fixed host. Deploy MLflow as its own Cloud
#     Run service or GKE deployment with a persistent DNS name instead.
#   - The always-on GPU VM (cifar10-train-vm) can then be decommissioned
#     entirely, removing its ongoing idle-time cost.