"""Test individual processing steps with synthetic data."""

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain.processing_steps import ManualOutlierRemovalStep, ResampleStep


@pytest.fixture
def sample_process_config():
    return {
        "process": {
            "resample": {
                "freq": "5min",
                "origin": "start_day",
                "label": "right",
                "closed": "right",
            },
            "manual_outliers": {
                "enabled": True,
                "panels": {
                    "1": {
                        "columns": {
                            "value": {"lower": 0, "upper": 100},
                        }
                    }
                },
            },
        }
    }


def test_resample_step(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=100, freq="1min", tz="UTC")
    df = pd.DataFrame({"value": np.arange(100, dtype=float)}, index=idx)

    step = ResampleStep(sample_process_config)
    result = step.process(df)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) < len(df)


def test_manual_outlier_removal(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
    values = np.arange(20, dtype=float) * 10  # 0, 10, 20, ..., 190
    values[5] = -999  # below lower bound
    values[15] = 999  # above upper bound
    df = pd.DataFrame({"value": values}, index=idx)

    step = ManualOutlierRemovalStep(sample_process_config, panel_id="1")
    result = step.process(df)

    assert pd.isna(result["value"].iloc[5])
    assert pd.isna(result["value"].iloc[15])
    assert not pd.isna(result["value"].iloc[10])
