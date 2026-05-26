"""Resample DataFrame to target frequency."""

from typing import Any, Optional

import pandas as pd

from spotanomaly2.domain.processing_steps.base import ProcessingStep


class ResampleStep(ProcessingStep):
    """Resample DataFrame to target frequency."""

    name = "Resampling data"

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

    def process(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
        resample_cfg = self.config["process"]["resample"]
        return df.resample(
            rule=resample_cfg["freq"],
            origin=resample_cfg["origin"],
            label=resample_cfg["label"],
            closed=resample_cfg["closed"],
        ).mean()
