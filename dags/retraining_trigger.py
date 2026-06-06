"""
Retraining Trigger DAG
Triggers the full retraining pipeline when called by the monitoring DAG.
Can also be triggered manually by an engineer.
"""

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

with DAG(
    dag_id='retraining_trigger',
    description='Triggers training and evaluation pipelines',
    default_args=default_args,
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['retraining', 'trigger']
) as dag:

    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training_pipeline',
        trigger_dag_id='training_pipeline',
        wait_for_completion=True,
        poke_interval=60,
    )

    trigger_evaluation = TriggerDagRunOperator(
        task_id='trigger_evaluation_pipeline',
        trigger_dag_id='evaluation_pipeline',
        wait_for_completion=True,
        poke_interval=60,
    )

    trigger_training >> trigger_evaluation