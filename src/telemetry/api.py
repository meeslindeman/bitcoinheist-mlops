from prometheus_client import Counter, Histogram

PREDICTION_REQUESTS = Counter(
    "prediction_requests",
    "Total number of prediction requests"
)

PREDICTION_ERRORS = Counter(
    "prediction_errors",
    "Total number of prediction errors"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests in seconds"
)

PREDICTION_LABELS = Counter(
    "prediction_labels",
    "Count of predictions by label",
    ["prediction"]
)