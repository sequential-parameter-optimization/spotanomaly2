"""Subtract a weather temperature baseline from the temperature signal."""

from typing import Any, Optional

import pandas as pd

from spotanomaly2.domain.processing_steps.base import ProcessingStep


class WeatherAdjustmentStep(ProcessingStep):
    """Subtract the merged ``weather_temperature_baseline`` from ``temperature``.

    Weather fetching and the rolling-baseline computation now live in
    ``WeatherFetcher.merge_into_panels`` (the merger runs once before
    ``DataProcessor``). This step is a pure per-panel transform: if both
    ``temperature`` and ``weather_temperature_baseline`` are present, subtract;
    otherwise pass through.
    """

    name = "Adjusting temperature with weather baseline"

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

    def process(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
        if "temperature" not in df.columns or "weather_temperature_baseline" not in df.columns:
            return df
        df = df.copy()
        df["temperature"] = df["temperature"] - df["weather_temperature_baseline"]
        if self.logger:
            self.logger.info(f"Adjusted temperature for {len(df)} timestamps using merged baseline")
        return df
