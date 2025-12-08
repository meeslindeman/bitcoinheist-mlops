import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

from prometheus_client import Gauge

from configs.configs import PathsConfig


def write_training_summary(summary: Dict[str, Any]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {"last_run_time": timestamp, **summary}

    path = Path(PathsConfig.telemetry_training_data_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_training_summary() -> Dict[str, Any] | None:
    path = Path(PathsConfig.telemetry_training_data_path)
    if not path.exists():
        return None

    with path.open("r") as f:
        return json.load(f)


TRAINING_LAST_RUN_TIMESTAMP = Gauge(
    "training_last_run_timestamp_seconds",
    "Unix timestamp of the last successful training run"
)

TRAINING_LAST_ACCURACY = Gauge(
    "training_last_accuracy",
    "Accuracy of the last training run (test set)"
)


TRAINING_LAST_ROC_AUC = Gauge(
    "training_last_roc_auc",
    "ROC AUC of the last training run (test set)"
)


TRAINING_LAST_F1_POSITIVE = Gauge(
    "training_last_f1_positive",
    "F1-score for the positive class of the last training run (test set)"
)


def init_training_metrics_from_file() -> None:
    summary = load_training_summary()
    if summary is None:
        return
    
    last_run_time = summary.get("last_run_time")
    if last_run_time:
        try:
            data = datetime.fromisoformat(last_run_time)
            TRAINING_LAST_RUN_TIMESTAMP.set(data.timestamp())
        except Exception:
            pass

    accuracy = summary.get("accuracy")
    if accuracy is not None:
        TRAINING_LAST_ACCURACY.set(float(accuracy))
    
    roc_auc = summary.get("roc_auc")
    if roc_auc is not None:
        TRAINING_LAST_ROC_AUC.set(float(roc_auc))

    f1_positive = summary.get("f1_positive")
    if f1_positive is not None:
        TRAINING_LAST_F1_POSITIVE.set(float(f1_positive))
