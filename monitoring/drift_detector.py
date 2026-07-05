"""
Drift Detector
Standalone module for detecting prediction drift in CIFAR-10 predictions
logged to BigQuery.

Usable two ways:
1. As a CLI script for an ad-hoc manual check:
     python3 -m monitoring.drift_detector
   Prints results to console and writes them to a timestamped JSON file
   under monitoring/results/.
2. Imported by monitoring_dag.py, so the DAG and this script share one
   source of truth for the actual detection logic rather than
   maintaining two copies.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

GCP_PROJECT = os.environ.get("GCP_PROJECT", "mlops-500119")
BQ_DATASET = os.environ.get("BQ_DATASET", "mlops_predictions")
BQ_TABLE = os.environ.get("BQ_TABLE", "predictions")

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Drift thresholds — same values used in monitoring_dag.py originally.
P_VALUE_THRESHOLD = 0.05
JS_DIVERGENCE_THRESHOLD = 0.1


def get_bigquery_client():
    from google.cloud import bigquery
    return bigquery.Client(project=GCP_PROJECT)


def check_prediction_volume(client=None):
    """
    Count predictions logged in the last 24 hours. Returns the count.
    """
    if client is None:
        client = get_bigquery_client()

    query = f"""
        SELECT COUNT(*) as count
        FROM `{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
    """
    result = list(client.query(query).result())
    count = result[0]['count']
    logger.info(f"Predictions in last 24 hours: {count}")
    return count


def detect_drift(client=None):
    """
    Detect drift by comparing the recent (last 24h) predicted-class
    distribution against a baseline (first week of predictions) using a
    Chi-squared test and Jensen-Shannon Divergence.

    Returns a dict with drift_detected, p_value, js_divergence, and the
    raw baseline/recent counts used, so results are fully inspectable
    rather than just a boolean.
    """
    from scipy import stats
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    if client is None:
        client = get_bigquery_client()

    baseline_query = f"""
        SELECT predicted_class, COUNT(*) as count
        FROM `{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE timestamp >= TIMESTAMP_SUB(
            (SELECT MIN(timestamp) FROM `{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`),
            INTERVAL 0 DAY
        )
        AND timestamp <= TIMESTAMP_ADD(
            (SELECT MIN(timestamp) FROM `{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`),
            INTERVAL 7 DAY
        )
        GROUP BY predicted_class
        ORDER BY predicted_class
    """

    recent_query = f"""
        SELECT predicted_class, COUNT(*) as count
        FROM `{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        GROUP BY predicted_class
        ORDER BY predicted_class
    """

    baseline_results = {row['predicted_class']: row['count']
                        for row in client.query(baseline_query).result()}
    recent_results = {row['predicted_class']: row['count']
                      for row in client.query(recent_query).result()}

    baseline = np.array([baseline_results.get(c, 0) for c in CLASSES], dtype=float)
    recent = np.array([recent_results.get(c, 0) for c in CLASSES], dtype=float)

    if baseline.sum() == 0 or recent.sum() == 0:
        logger.warning("Insufficient data for drift detection")
        return {
            'drift_detected': False,
            'p_value': None,
            'js_divergence': None,
            'baseline_counts': baseline_results,
            'recent_counts': recent_results,
            'reason': 'insufficient_data',
        }

    baseline_prop = baseline / baseline.sum()
    recent_prop = recent / recent.sum()

    expected = baseline_prop * recent.sum()

    # scipy's chisquare divides by the expected value for each category.
    # Any class with zero baseline representation (expected == 0) causes
    # a 0/0 division if that same class also has zero recent occurrences
    # — common with sparse data, e.g. only a handful of predictions total
    # so most of the 10 classes are unrepresented in both windows. This
    # produces NaN for the whole statistic, not just that one class,
    # since NaN propagates through the sum. Fix: only compare classes
    # with nonzero expected counts.
    nonzero_mask = expected > 0
    if nonzero_mask.sum() < 2:
        logger.warning(
            "Fewer than 2 classes with nonzero baseline representation — "
            "not enough categories for a meaningful chi-squared test"
        )
        return {
            'drift_detected': False,
            'p_value': None,
            'js_divergence': float(jensenshannon(baseline_prop, recent_prop)),
            'baseline_counts': baseline_results,
            'recent_counts': recent_results,
            'reason': 'insufficient_class_coverage',
        }

    chi2, p_value = stats.chisquare(
        recent[nonzero_mask], f_exp=expected[nonzero_mask]
    )
    js_divergence = jensenshannon(baseline_prop, recent_prop)

    logger.info(f"Chi-squared: {chi2:.4f}, p-value: {p_value:.4f}")
    logger.info(f"Jensen-Shannon Divergence: {js_divergence:.4f}")

    drift_detected = (
        p_value < P_VALUE_THRESHOLD or js_divergence > JS_DIVERGENCE_THRESHOLD
    )
    logger.info(f"Drift detected: {drift_detected}")

    return {
        'drift_detected': bool(drift_detected),
        'p_value': float(p_value),
        'js_divergence': float(js_divergence),
        'baseline_counts': baseline_results,
        'recent_counts': recent_results,
        'reason': None,
    }


def run_check(output_dir='monitoring/results'):
    """
    Run the full volume + drift check, print a summary, and write results
    to a timestamped JSON file. Returns the combined result dict.
    """
    client = get_bigquery_client()

    prediction_count = check_prediction_volume(client)
    drift_result = detect_drift(client)

    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        'checked_at': timestamp,
        'prediction_count_24h': prediction_count,
        **drift_result,
    }

    print(json.dumps(result, indent=2))

    os.makedirs(output_dir, exist_ok=True)
    filename = f"drift_check_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {output_path}")

    return result


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Run a manual drift check')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='monitoring/results',
        help='Directory to write the JSON result file to'
    )
    args = parser.parse_args()
    run_check(output_dir=args.output_dir)


if __name__ == '__main__':
    main()