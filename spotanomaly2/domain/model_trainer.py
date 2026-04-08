"""Model training service using spotforecast2."""

from typing import Any

import pandas as pd

from spotanomaly2.domain.spotforecast_adapter import SpotforecastTrainer
from spotanomaly2.infrastructure import logging


class ModelTrainer:
    """Delegates training to SpotforecastTrainer (spotforecast2)."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("ModelTrainer")
        self.trainer = SpotforecastTrainer(config, self.logger)

    def train_panel(
        self,
        panel_id: str,
        df: pd.DataFrame,
        timestamp: str = None,
        save_model: bool = True,
        panel_specific_params: dict = None,
        channel_specific_params: dict = None,
    ) -> tuple[pd.DataFrame, str]:
        return self.trainer.train_panel(
            panel_id,
            df,
            timestamp=timestamp,
            save_model=save_model,
            panel_specific_params=panel_specific_params,
            channel_specific_params=channel_specific_params,
        )

    def train_all_panels(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        return self.trainer.train_all_panels(panel_data)

    def run(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        return self.trainer.run(panel_data)
