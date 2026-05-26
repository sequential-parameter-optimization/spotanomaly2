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

        # Weather is fetched + merged upstream by PanelDataMerger; WeatherAdjustmentStep
        # is a pure transform that consumes the pre-merged weather_temperature_baseline.
        # ManualOutlierRemovalStep receives panel_id at call time via process_panel.
        self._steps = [
            ResampleStep(config, self.logger),
            MaintenanceRemovalStep(config, self.logger),
            ManualOutlierRemovalStep(config, self.logger),
            ImputationStep(config, self.logger),
            MedianFilterStep(config, self.logger),
            TemperatureAggregationStep(config, self.logger),
            WeatherAdjustmentStep(config, self.logger),
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
        for step in self._steps:
            self.logger.info(f"Step: {step.name}")
            df = step.process(df, panel_id=panel_id)

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
