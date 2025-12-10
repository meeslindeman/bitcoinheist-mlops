import time
from typing import Any, Dict

from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

from configs.configs import TelemetryConfig


def push_training_summary(summary: Dict[str, Any]) -> None:
    registry = CollectorRegistry()

    last_run_ts = Gauge(
        "training_last_run_timestamp_seconds",
        "Unix timestamp of the last successful training run",
        registry=registry,
    )

    last_accuracy = Gauge(
        "training_last_accuracy",
        "Accuracy of the last training run (test set)",
        registry=registry,
    )

    last_roc_auc = Gauge(
        "training_last_roc_auc",
        "ROC AUC of the last training run (test set)",
        registry=registry,
    )

    last_f1_positive = Gauge(
        "training_last_f1_positive",
        "F1-score for the positive class of the last training run (test set)",
        registry=registry,
    )

    last_run_ts.set(time.time())

    accuracy = summary.get("accuracy")
    if accuracy is not None:
        last_accuracy.set(float(accuracy))

    roc_auc = summary.get("roc_auc")
    if roc_auc is not None:
        last_roc_auc.set(float(roc_auc))

    f1_positive = summary.get("f1_positive")
    if f1_positive is not None:
        last_f1_positive.set(float(f1_positive))
        
    push_to_gateway(
        TelemetryConfig.push_gateway_uri,
        job="training",
        registry=registry,
    )