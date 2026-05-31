"""Model training service using spotforecast2."""

from pathlib import Path
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
        panel_data: pd.DataFrame,
        timestamp: str,
        save_model: bool = True,
    ) -> tuple[pd.DataFrame, str]:
        eval_df, timestamp = self.trainer.train_panel(
            panel_id,
            panel_data,
            timestamp=timestamp,
            save_model=save_model,
        )

        base_models_dir = Path(self.config["paths"]["models_dir"])
        models_dir = base_models_dir / str(timestamp)
        eval_filename = f"fc_model_panel_{panel_id}_eval.csv"
        eval_path = models_dir / eval_filename
        eval_df.to_csv(eval_path)
        self.logger.info(f"Saved evaluation results to {eval_path}")

        return eval_df, timestamp

    def train_all_panels(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        return self.trainer.train_all_panels(panel_data)

    def run(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        return self.trainer.run(panel_data)
