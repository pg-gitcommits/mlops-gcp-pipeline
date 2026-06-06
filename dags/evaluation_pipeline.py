"""
Evaluation Pipeline DAG
Evaluates trained model checkpoints and generates reports.
Triggered after training_pipeline completes.
"""

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

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
    description='Evaluate CIFAR-10 model and generate reports',
    default_args=default_args,
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['evaluation', 'cifar10']
) as dag:

    evaluate_task = KubernetesPodOperator(
        task_id='evaluate_model',
        name='cifar10-evaluation-pod',
        image=IMAGE,
        cmds=['python3', '-m', 'src.evaluate', '--config', 'configs/train_config.yaml'],
        env_vars={
            'GCP_PROJECT_ID': 'mlops-50050',
            'GCS_BUCKET': 'mlops-cifar10-artifacts',
        },
        get_logs=True,
        is_delete_operator_pod=True,
        in_cluster=True,
    )

    upload_logs_task = KubernetesPodOperator(
        task_id='upload_logs_to_gcs',
        name='cifar10-upload-logs-pod',
        image=IMAGE,
        cmds=[
            'python3', '-c',
            '''
import subprocess
subprocess.run([
    "gsutil", "-m", "cp", "-r",
    "logs/",
    "gs://mlops-cifar10-artifacts/evaluation-logs/"
])
print("Logs uploaded to GCS")
'''
        ],
        env_vars={
            'GCP_PROJECT_ID': 'mlops-50050',
        },
        get_logs=True,
        is_delete_operator_pod=True,
        in_cluster=True,
    )

    evaluate_task >> upload_logs_task