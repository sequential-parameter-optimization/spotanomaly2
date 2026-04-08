"""Data processing service combining convert, resample, and preprocess steps."""

from typing import Any, Optional

import pandas as pd

from spotanomaly2.domain.processing_steps import (
    ImputationStep,
    MaintenanceRemovalStep,
    ManualOutlierRemovalStep,
    MedianFilterStep,
    ResampleStep,
    TemperatureAggregationStep,
    WeatherAdjustmentStep,
)
from spotanomaly2.domain.weather_fetcher import WeatherFetcher
from spotanomaly2.infrastructure import logging


class DataProcessor:
    """Orchestrates processing steps: resample, maintenance, imputation, filter, temperature, weather."""

    def __init__(self, config: dict[str, Any], logger=None):
        """Initialize DataProcessor with configuration.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.get_logger("DataProcessor")

        weather_fetcher: Optional[WeatherFetcher] = None
        weather_cfg = self.config.get("process", {}).get("weather", {})
        if weather_cfg.get("enabled", False):
            weather_config = {
                "latitude": weather_cfg.get("latitude"),
                "longitude": weather_cfg.get("longitude"),
                "use_forecast": weather_cfg.get("use_forecast", True),
            }
            if weather_config["latitude"] is None or weather_config["longitude"] is None:
                self.logger.warning(
                    "Weather adjustment enabled but latitude/longitude not configured. "
                    "Weather adjustment will be skipped."
                )
            else:
                weather_fetcher = WeatherFetcher(weather_config, self.logger)

        # Initialize processing steps with panel_id placeholder (will be set per panel)
        self._base_steps = [
            ResampleStep(config, self.logger),
            MaintenanceRemovalStep(config, self.logger),
        ]
        self._post_outlier_steps = [
            ImputationStep(config, self.logger),
            MedianFilterStep(config, self.logger),
            TemperatureAggregationStep(config, self.logger),
            WeatherAdjustmentStep(config, self.logger, weather_fetcher),
        ]

    def process_panel(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
        """Process a single panel through the full pipeline.

        Pipeline: resample → maintenance removal → manual outlier removal → imputation
                  → median filter → temperature aggregation → weather adjustment.


        Args:
            df: Raw panel DataFrame
            panel_id: Optional panel identifier for panel-specific processing

        Returns:
            Processed DataFrame
        """
        # Build complete step list with panel-specific manual outlier removal
        steps = (
            self._base_steps + [ManualOutlierRemovalStep(self.config, self.logger, panel_id)] + self._post_outlier_steps
        )

        step_names = [
            "Resampling data",
            "Removing maintenance periods",
            "Manual outlier removal",
            "Imputing missing values",
            "Applying median filter to flow columns",
            "Aggregating temperature sensors",
            "Adjusting temperature with weather baseline",
        ]

        for step, name in zip(steps, step_names):
            self.logger.info(f"Step: {name}")
            df = step.process(df)

        return df

    def process_all_panels(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Process all panels from raw to processed.

        Args:
            panel_data: Dictionary mapping panel_id to raw DataFrame

        Returns:
            Dictionary mapping panel_id to processed DataFrame
        """
        processed_data = {}
        for panel_id, df_raw in panel_data.items():
            self.logger.info(f"Processing panel {panel_id}...")
            self.logger.info(f"Processing {len(df_raw)} rows for panel {panel_id}")
            df_processed = self.process_panel(df_raw, panel_id)
            processed_data[panel_id] = df_processed
            self.logger.info(f"Processed panel {panel_id} with {len(df_processed)} rows")
        return processed_data

    def run(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Run the complete processing pipeline.

        Args:
            panel_data: Dictionary mapping panel_id to raw DataFrame

        Returns:
            Dictionary mapping panel_id to processed DataFrame
        """
        self.logger.info("Starting data processing...")
        processed_data = self.process_all_panels(panel_data)
        self.logger.info("Data processing completed successfully")
        return processed_data
