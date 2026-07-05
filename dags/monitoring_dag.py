"""
Monitoring DAG
Runs daily to detect data drift in prediction logs.
Triggers retraining automatically if drift is detected.

Drift detection logic lives in monitoring/drift_detector.py, shared with
the standalone CLI script — this DAG is a thin wrapper around it, not a
second copy of the same logic.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

from monitoring.drift_detector import check_prediction_volume, detect_drift

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}


def check_prediction_volume_task(**context):
    """Check if enough predictions exist for drift analysis."""
    count = check_prediction_volume()
    context['ti'].xcom_push(key='prediction_count', value=count)
    return count


def detect_drift_task(**context):
    """Run drift detection and push results to XCom."""
    result = detect_drift()
    context['ti'].xcom_push(key='drift_detected', value=result['drift_detected'])
    context['ti'].xcom_push(key='p_value', value=result['p_value'])
    context['ti'].xcom_push(key='js_divergence', value=result['js_divergence'])
    return result['drift_detected']


def branch_on_drift(**context):
    """Branch DAG based on drift detection result."""
    drift_detected = context['ti'].xcom_pull(
        task_ids='detect_drift',
        key='drift_detected'
    )
    if drift_detected:
        logger.info("Drift detected — triggering retraining")
        return 'trigger_retraining'
    else:
        logger.info("No drift detected — skipping retraining")
        return 'no_drift'


def log_drift_results(**context):
    """Log drift detection results to BigQuery."""
    drift_detected = context['ti'].xcom_pull(
        task_ids='detect_drift', key='drift_detected'
    )
    p_value = context['ti'].xcom_pull(
        task_ids='detect_drift', key='p_value'
    )
    js_divergence = context['ti'].xcom_pull(
        task_ids='detect_drift', key='js_divergence'
    )

    logger.info(
        f"Drift results — detected: {drift_detected}, "
        f"p_value: {p_value}, js_divergence: {js_divergence}"
    )


with DAG(
    dag_id='monitoring_dag',
    description='Daily drift detection and automated retraining trigger',
    default_args=default_args,
    schedule='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['monitoring', 'drift', 'cifar10']
) as dag:

    check_volume_task = PythonOperator(
        task_id='check_prediction_volume',
        python_callable=check_prediction_volume_task,
        provide_context=True
    )

    detect_drift_task_op = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift_task,
        provide_context=True
    )

    branch_task = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=branch_on_drift,
        provide_context=True
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id='trigger_retraining',
        trigger_dag_id='retraining_trigger',
        wait_for_completion=False,
    )

    no_drift = EmptyOperator(
        task_id='no_drift'
    )

    log_results = PythonOperator(
        task_id='log_drift_results',
        python_callable=log_drift_results,
        provide_context=True,
        trigger_rule='none_failed_min_one_success'
    )

    check_volume_task >> detect_drift_task_op >> branch_task
    branch_task >> trigger_retraining >> log_results
    branch_task >> no_drift >> log_results
