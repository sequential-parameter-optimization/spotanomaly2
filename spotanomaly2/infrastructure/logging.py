"""Logging configuration and setup."""

import logging


def get_logger(name: str = "eventdetection") -> logging.Logger:
    """Get a configured logger instance."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(name)
