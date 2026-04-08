"""Custom exception hierarchy for the event detection domain."""


class SpotAnomalyError(Exception):
    """Base exception for all domain errors."""

    pass


class ModelNotFoundException(SpotAnomalyError, FileNotFoundError):
    """Raised when a required model file cannot be found."""

    pass


class InsufficientDataException(SpotAnomalyError, ValueError):
    """Raised when there is not enough data for training or detection."""

    pass


class ConfigurationException(SpotAnomalyError, ValueError):
    """Raised when configuration is invalid or missing required values."""

    pass
