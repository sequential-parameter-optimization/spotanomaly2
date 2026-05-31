"""Readiness checks for pipeline baseline data and trained models."""

from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from spotanomaly2.infrastructure import logging, storage


class ReadinessChecker:
    """Checks whether baseline processed data and trained models are present and fresh."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("ReadinessChecker")

    def is_ready(self, max_age_days: float = 7) -> bool:
        """Check whether baseline data and trained models exist and are fresh.

        Args:
            max_age_days: Maximum age (in days) of processed data and models
                before they are considered stale. Defaults to 7.

        Returns True when every configured panel has processed data, at least
        one trained model exists on disk, and both the data and model are
        newer than *max_age_days*.
        """
        return self.data_is_ready(max_age_days) and self.model_is_ready(max_age_days)

    def data_is_ready(self, max_age_days: float = 7) -> bool:
        """Return True iff every configured panel has fresh processed data.

        Args:
            max_age_days: Maximum age (in days) of the processed data before it
                is considered stale. Defaults to 7.
        """
        processed_dir = Path(self.config["paths"]["processed_dir"])
        panels = self.config["panels"]["panel_ids"]

        now = pd.Timestamp.now(tz="UTC")
        max_age = timedelta(days=max_age_days)

        return all(self._is_panel_data_ready(panel_id, processed_dir, now, max_age) for panel_id in panels)

    def model_is_ready(self, max_age_days: float = 7) -> bool:
        """Return True iff at least one fresh trained model exists on disk.

        Args:
            max_age_days: Maximum age (in days) of the trained model before it
                is considered stale. Defaults to 7.
        """
        models_dir = Path(self.config["paths"]["models_dir"])

        now = pd.Timestamp.now(tz="UTC")
        max_age = timedelta(days=max_age_days)

        return self._is_model_ready(models_dir, now, max_age)

    @staticmethod
    def _is_panel_data_ready(
        panel_id: str,
        processed_dir: Path,
        now: pd.Timestamp,
        max_age: timedelta,
    ) -> bool:
        """True if the panel has non-empty processed data with a recent latest timestamp."""
        panel_file = processed_dir / f"panel_{panel_id}.parquet"
        try:
            df = storage.load_parquet(panel_file)
        except FileNotFoundError:
            return False
        if df.empty:
            return False
        latest_ts = pd.Timestamp(df.index.max())
        if latest_ts.tz is None:
            latest_ts = latest_ts.tz_localize("UTC")
        return (now - latest_ts) <= max_age

    @staticmethod
    def _is_model_ready(
        models_dir: Path,
        now: pd.Timestamp,
        max_age: timedelta,
    ) -> bool:
        """True if the latest timestamped model dir contains a recent trained model."""
        if not models_dir.exists():
            return False
        try:
            latest_dir = storage.find_latest_timestamped_dir(models_dir)
        except FileNotFoundError:
            return False
        model_files = list(latest_dir.glob("fc_model_panel_*.pkl"))
        if not model_files:
            return False
        oldest_model = min(f.stat().st_mtime for f in model_files)
        model_age = now - pd.Timestamp(oldest_model, unit="s", tz="UTC")
        return model_age <= max_age
