"""Application-wide constants for training, detection, and server configuration."""

# Training / detection thresholds
MIN_TRAIN_SIZE = 100  # Minimum viable training size for anomaly detection
MIN_ROWS_FOR_DETECTION = 200  # Minimum rows required for detection (e.g. in live mode)
TRAIN_TEST_SPLIT_RATIO = 0.6  # Fraction of data used for training when adjusting for insufficient data

# Live report server
LIVE_REPORT_SERVER_PORT = 8050
FILE_CHANGE_DEBOUNCE_SECONDS = 1.0  # Debounce file watcher to avoid rapid duplicate events
