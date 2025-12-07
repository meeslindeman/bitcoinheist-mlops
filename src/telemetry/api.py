from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

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

def metrics_response():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
