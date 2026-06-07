"""
Training Pipeline DAG
Orchestrates CIFAR-10 model training using GKE.
Triggered manually or by the retraining trigger DAG.
"""

from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.providers.google.cloud.operators.kubernetes_engine import (
    GKEStartPodOperator
)

default_args = {
    'owner': 'mlops',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

IMAGE = 'europe-west4-docker.pkg.dev/mlops-50050/mlops-images/cifar10-train:latest'

with DAG(
    dag_id='training_pipeline',
    description='Train ResNet-18 on CIFAR-10 using GKE',
    default_args=default_args,
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['training', 'cifar10', 'resnet18']
) as dag:

    train_task = GKEStartPodOperator(
        task_id='train_model',
        project_id='mlops-50050',
        location='europe-west4-a',
        cluster_name='cifar10-cluster',
        name='cifar10-training-pod',
        image=IMAGE,
        cmds=['python3', '-m', 'src.train', '--config', 'configs/train_config.yaml'],
        env_vars={
            'GCP_PROJECT_ID': 'mlops-50050',
            'GCS_BUCKET': 'mlops-cifar10-artifacts',
        },
        get_logs=True,
        is_delete_operator_pod=True,
    )