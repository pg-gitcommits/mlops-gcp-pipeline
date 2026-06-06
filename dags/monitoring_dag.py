"""
Monitoring DAG
Runs daily to detect data drift in prediction logs.
Triggers retraining automatically if drift is detected.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}


def check_prediction_volume(**context):
    """Check if enough predictions exist for drift analysis."""
    from google.cloud import bigquery

    client = bigquery.Client(project='mlops-50050')
    query = """
        SELECT COUNT(*) as count
        FROM `mlops-50050.mlops_predictions.predictions`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
    """
    result = list(client.query(query).result())
    count = result[0]['count']
    logger.info(f"Predictions in last 24 hours: {count}")
    context['ti'].xcom_push(key='prediction_count', value=count)
    return count


def detect_drift(**context):
    """
    Detect drift by comparing recent prediction distribution
    against baseline distribution using Chi-squared test.
    """
    from google.cloud import bigquery
    from scipy import stats
    import numpy as np

    client = bigquery.Client(project='mlops-50050')

    # Get baseline distribution (first week of predictions)
    baseline_query = """
        SELECT predicted_class, COUNT(*) as count
        FROM `mlops-50050.mlops_predictions.predictions`
        WHERE timestamp >= TIMESTAMP_SUB(
            (SELECT MIN(timestamp) FROM `mlops-50050.mlops_predictions.predictions`),
            INTERVAL 0 DAY
        )
        AND timestamp <= TIMESTAMP_ADD(
            (SELECT MIN(timestamp) FROM `mlops-50050.mlops_predictions.predictions`),
            INTERVAL 7 DAY
        )
        GROUP BY predicted_class
        ORDER BY predicted_class
    """

    # Get recent distribution (last 24 hours)
    recent_query = """
        SELECT predicted_class, COUNT(*) as count
        FROM `mlops-50050.mlops_predictions.predictions`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        GROUP BY predicted_class
        ORDER BY predicted_class
    """

    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    baseline_results = {row['predicted_class']: row['count']
                        for row in client.query(baseline_query).result()}
    recent_results = {row['predicted_class']: row['count']
                      for row in client.query(recent_query).result()}

    # Build distribution arrays
    baseline = np.array([baseline_results.get(c, 0) for c in classes], dtype=float)
    recent = np.array([recent_results.get(c, 0) for c in classes], dtype=float)

    # Avoid division by zero
    if baseline.sum() == 0 or recent.sum() == 0:
        logger.warning("Insufficient data for drift detection")
        context['ti'].xcom_push(key='drift_detected', value=False)
        return False

    # Normalise to proportions
    baseline_prop = baseline / baseline.sum()
    recent_prop = recent / recent.sum()

    # Chi-squared test
    expected = baseline_prop * recent.sum()
    chi2, p_value = stats.chisquare(recent, f_exp=expected)

    # Jensen-Shannon Divergence
    from scipy.spatial.distance import jensenshannon
    js_divergence = jensenshannon(baseline_prop, recent_prop)

    logger.info(f"Chi-squared: {chi2:.4f}, p-value: {p_value:.4f}")
    logger.info(f"Jensen-Shannon Divergence: {js_divergence:.4f}")

    # Drift detected if p-value < 0.05 OR JS divergence > 0.1
    drift_detected = p_value < 0.05 or js_divergence > 0.1

    logger.info(f"Drift detected: {drift_detected}")

    context['ti'].xcom_push(key='drift_detected', value=drift_detected)
    context['ti'].xcom_push(key='p_value', value=float(p_value))
    context['ti'].xcom_push(key='js_divergence', value=float(js_divergence))

    return drift_detected


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
    from google.cloud import bigquery
    from datetime import datetime

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
        python_callable=check_prediction_volume,
        provide_context=True
    )

    detect_drift_task = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift,
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

    check_volume_task >> detect_drift_task >> branch_task
    branch_task >> trigger_retraining >> log_results
    branch_task >> no_drift >> log_results