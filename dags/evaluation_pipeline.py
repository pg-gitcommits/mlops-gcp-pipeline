"""
Evaluation Pipeline DAG
Evaluates the CIFAR-10 model on the dedicated GPU VM (cifar10-train-vm).
Triggered after training_pipeline completes.

Same architecture as training_pipeline.py — SSH to the always-on GPU VM
rather than a GKE pod. See training_pipeline.py for the full rationale
and the production-scale note on what changes at higher GPU quota.

Requires the same Airflow SSH Connection as training_pipeline.py:
'cifar10_vm_ssh'.
"""

from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

SSH_CONN_ID = 'cifar10_vm_ssh'
REPO_DIR = '~/mlops-gcp-pipeline'
GCS_BUCKET = 'mlops-500119-cifar10-artifacts'

with DAG(
    dag_id='evaluation_pipeline',
    description='Evaluate CIFAR-10 model on the GPU VM',
    default_args=default_args,
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['evaluation', 'cifar10']
) as dag:

    evaluate_task = SSHOperator(
        task_id='evaluate_model',
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f'bash -c "cd {REPO_DIR} && '
            'source venv/bin/activate && '
            'python3 -m src.evaluate --config configs/train_config.yaml"'
        ),
        cmd_timeout=1800,
    )

    upload_logs_task = SSHOperator(
        task_id='upload_logs_to_gcs',
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f'bash -c "cd {REPO_DIR} && '
            f'gsutil -m cp -r logs/ gs://{GCS_BUCKET}/evaluation-logs/"'
        ),
        cmd_timeout=600,
    )

    evaluate_task >> upload_logs_task