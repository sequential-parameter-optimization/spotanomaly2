"""Average all temperature sensors into a single 'temperature' column."""

from typing import Any, Optional

import pandas as pd

from spotanomaly2.domain.processing_steps.base import ProcessingStep


class TemperatureAggregationStep(ProcessingStep):
    """Average all temperature sensors into a single 'temperature' column."""

    name = "Aggregating temperature sensors"

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.weight_suffix = self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")

    def process(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
        temp_pattern = self.config["process"]["temperature_columns_pattern"]
        temp_cols = [col for col in df.columns if temp_pattern in col and not col.endswith(self.weight_suffix)]
        if not temp_cols:
            if self.logger:
                self.logger.warning("No temperature columns found to aggregate")
            return df
        df = df.copy()
        df["temperature"] = df[temp_cols].mean(axis=1)
        df.drop(columns=temp_cols, inplace=True)
        return df
