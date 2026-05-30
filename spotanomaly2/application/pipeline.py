"""Pipeline orchestration for event detection system."""

from typing import Any

import pandas as pd

from spotanomaly2.application.data_manager import DataManager
from spotanomaly2.application.exogenous_downloader import ExogenousDownloader
from spotanomaly2.application.exogenous_joiner import ExogenousJoiner
from spotanomaly2.application.readiness_checker import ReadinessChecker
from spotanomaly2.domain.anomaly_detector import AnomalyDetector
from spotanomaly2.domain.data_processor import DataProcessor
from spotanomaly2.domain.model_trainer import ModelTrainer
from spotanomaly2.domain.primary_fetcher import PrimaryDataFetcher
from spotanomaly2.infrastructure import logging
from spotanomaly2.infrastructure.storage import generate_timestamp


class Pipeline:
    """Orchestrates the event detection pipeline (download, process, train, detect)."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("Pipeline")
        self._data_manager = DataManager(config, self.logger)
        self._readiness_checker = ReadinessChecker(config, self.logger)
        self._exogenous_downloader = ExogenousDownloader(config, self.logger)
        self._exogenous_joiner = ExogenousJoiner(config, self.logger)

    # ------------------------------------------------------------------
    # Readiness checks
    # ------------------------------------------------------------------

    def data_is_ready(self, max_age_days: float = 7) -> bool:
        """Return True iff every configured panel has fresh processed data."""
        return self._readiness_checker.data_is_ready(max_age_days)

    def model_is_ready(self, max_age_days: float = 7) -> bool:
        """Return True iff at least one fresh trained model is present."""
        return self._readiness_checker.model_is_ready(max_age_days)

    # ------------------------------------------------------------------
    # Individual pipeline steps
    # ------------------------------------------------------------------

    def download(self, ignore_cache: bool = False) -> dict[str, pd.DataFrame]:
        """Download primary panel data and exogenous data.

        Args:
            ignore_cache: When True, bypass incremental gap-filling and re-fetch
                the full configured window for both primary and exogenous
                sources, overwriting any cached data. When False (default),
                only missing slices are fetched.

        Returns:
            Dict mapping panel_id to the raw panel DataFrame.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP: Download data from API")
        self.logger.info("=" * 60)

        primary_fetcher = PrimaryDataFetcher(self.config, self.logger)
        panel_data = primary_fetcher.run(ignore_cache=ignore_cache)
        self._data_manager.save_raw_data(panel_data)

        start, end = self._exogenous_downloader.derive_fetch_window(panel_data)
        self._exogenous_downloader.download_all(start, end, ignore_cache=ignore_cache)

        return panel_data

    def process(self) -> dict[str, pd.DataFrame]:
        """Join exogenous features, resample and impute, then persist.

        Returns:
            Dict mapping panel_id to the processed panel DataFrame.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP: Process data")
        self.logger.info("=" * 60)

        panel_data = self._data_manager.load_raw_data()
        panel_data = self._exogenous_joiner.join_all(panel_data)

        processor = DataProcessor(self.config, self.logger)
        processed_data = processor.run(panel_data)
        self._data_manager.save_processed_data(processed_data)

        return processed_data

    def train(self) -> dict[str, tuple[pd.DataFrame, str]]:
        """Train one forecasting model per panel and persist them.

        Returns:
            Dict mapping panel_id to ``(evaluation_df, model_timestamp)``.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP: Train forecasting models")
        self.logger.info("=" * 60)

        panel_data = self._data_manager.load_processed_data()
        training_timestamp = generate_timestamp()

        trainer = ModelTrainer(self.config, self.logger)

        results: dict[str, tuple[pd.DataFrame, str]] = {}
        for panel_id, df in panel_data.items():
            eval_df, timestamp = trainer.train_panel(panel_id, df, timestamp=training_timestamp, save_model=True)
            results[panel_id] = (eval_df, timestamp)

        return results

    def tune(
        self,
        panel_id: str | None = None,
        channel: str | None = None,
    ) -> dict:
        """Tune forecaster hyperparameters per channel per panel via SpotOptim.

        Args:
            panel_id: If provided, tune only this panel. Otherwise tune all panels.
            channel: If provided, tune only this channel within selected panel(s).

        Returns:
            Dict of panel_id -> channel -> tuning results.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP: Tune forecaster hyperparameters (SpotOptim)")
        self.logger.info("=" * 60)

        # Lazy import: spotoptim / surrogate-model deps are heavy and
        # detect-only runs shouldn't pay for them.
        from spotanomaly2.domain.model_tuner import ModelTuner

        panel_data = self._data_manager.load_processed_data()

        if panel_id is not None:
            if panel_id not in panel_data:
                raise ValueError(f"Panel {panel_id} not found. Available: {list(panel_data.keys())}")
            panel_data = {panel_id: panel_data[panel_id]}

        channels_filter = [channel] if channel else None
        return ModelTuner(self.config, self.logger).run(panel_data, channels=channels_filter)

    def detect(self) -> dict[str, tuple]:
        """Run anomaly detection over processed data and persist results.

        Returns:
            Dict mapping panel_id to the detection result tuple.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP: Detect anomalies")
        self.logger.info("=" * 60)

        panel_data = self._data_manager.load_processed_data()
        detector = AnomalyDetector(self.config, self.logger)
        results = detector.run(panel_data)
        self._data_manager.save_detection_results(results)

        return results
