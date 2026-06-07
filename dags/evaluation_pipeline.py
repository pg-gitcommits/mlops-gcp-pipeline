"""
Evaluation Pipeline DAG
Evaluates CIFAR-10 model using GKE.
Triggered after training_pipeline completes.
"""

from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.providers.google.cloud.operators.kubernetes_engine import (
    GKEStartPodOperator
)

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

IMAGE = 'europe-west4-docker.pkg.dev/mlops-50050/mlops-images/cifar10-train:latest'

with DAG(
    dag_id='evaluation_pipeline',
    description='Evaluate CIFAR-10 model using GKE',
    default_args=default_args,
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['evaluation', 'cifar10']
) as dag:

    evaluate_task = GKEStartPodOperator(
        task_id='evaluate_model',
        project_id='mlops-50050',
        location='europe-west4-a',
        cluster_name='cifar10-cluster',
        name='cifar10-evaluation-pod',
        image=IMAGE,
        cmds=['python3', '-m', 'src.evaluate', '--config', 'configs/train_config.yaml'],
        env_vars={
            'GCP_PROJECT_ID': 'mlops-50050',
            'GCS_BUCKET': 'mlops-cifar10-artifacts',
        },
        get_logs=True,
        is_delete_operator_pod=True,
    )

    upload_logs_task = GKEStartPodOperator(
        task_id='upload_logs_to_gcs',
        project_id='mlops-50050',
        location='europe-west4-a',
        cluster_name='cifar10-cluster',
        name='cifar10-upload-logs-pod',
        image=IMAGE,
        cmds=[
            'python3', '-c',
            'import subprocess; subprocess.run(["gsutil", "-m", "cp", "-r", "logs/", "gs://mlops-cifar10-artifacts/evaluation-logs/"])'
        ],
        env_vars={
            'GCP_PROJECT_ID': 'mlops-50050',
        },
        get_logs=True,
        is_delete_operator_pod=True,
    )

    evaluate_task >> upload_logs_task