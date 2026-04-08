"""Model tuning service using spotforecast2 SpotOptim."""

from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from spotanomaly2.domain.spotforecast_adapter import SpotforecastTuner
from spotanomaly2.infrastructure import logging


class ModelTuner:
    """Merges global/panel/channel tune config and delegates to SpotforecastTuner."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("ModelTuner")
        self.tuner = SpotforecastTuner(config, self.logger)

    def _build_tune_config(self, panel_id: str | None = None) -> dict[str, Any]:
        """Build the effective tune config, merging global defaults with panel overrides."""
        tune_section = self.config.get("tune", {})
        return {
            "n_trials": tune_section.get("n_trials", 10),
            "n_initial": tune_section.get("n_initial", 5),
            "metric": tune_section.get("metric", "mean_absolute_error"),
            "search_space": dict(tune_section.get("search_space", {})),
            "panel_overrides": tune_section.get("panel_overrides", {}),
            "models": tune_section.get("models", ["LightGBM"]),
        }

    def tune_panel(
        self,
        panel_id: str,
        df: pd.DataFrame,
        channels: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        tune_config = self._build_tune_config(panel_id)
        return self.tuner.tune_panel(panel_id, df, tune_config, channels=channels)

    def tune_all_panels(
        self,
        panel_data: dict[str, pd.DataFrame],
        channels: list[str] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        results = {}
        panel_ids = list(panel_data.keys())
        panel_pbar = tqdm(panel_ids, desc="Panels", unit="panel", leave=True)
        for panel_id in panel_pbar:
            panel_pbar.set_postfix_str(f"panel {panel_id}")
            results[panel_id] = self.tune_panel(panel_id, panel_data[panel_id], channels=channels)
        panel_pbar.close()
        return results
