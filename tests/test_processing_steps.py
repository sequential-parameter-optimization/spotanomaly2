# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test individual processing steps with synthetic data."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain.processing_steps import (
    ImputationStep,
    MaintenanceRemovalStep,
    ManualOutlierRemovalStep,
    MedianFilterStep,
    ResampleStep,
    TemperatureAggregationStep,
    WeatherAdjustmentStep,
)


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
            "maintenance_column": "maintenance_flag",
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
            "imputation": {
                "method": "linear_interpolation",
                "params": {},
                "add_weights": True,
                "weight_suffix": "__weight",
            },
            "flow_columns_pattern": "flow",
            "temperature_columns_pattern": "temperature",
            "median_filter_kernel": 3,
            "weather": {"enabled": False},
        }
    }


# ----- existing tests --------------------------------------------------------


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

    step = ManualOutlierRemovalStep(sample_process_config)
    result = step.process(df, panel_id="1")

    assert pd.isna(result["value"].iloc[5])
    assert pd.isna(result["value"].iloc[15])
    assert not pd.isna(result["value"].iloc[10])


# ----- MaintenanceRemovalStep -----------------------------------------------


def test_maintenance_removal_all_false_unchanged(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "value": np.arange(10, dtype=float),
            "maintenance_flag": np.zeros(10, dtype=int),
        },
        index=idx,
    )
    step = MaintenanceRemovalStep(sample_process_config)
    result = step.process(df)
    # maintenance_flag column was dropped
    assert "maintenance_flag" not in result.columns
    # No values were masked
    assert not result["value"].isna().any()


def test_maintenance_removal_all_true_sets_nan(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "value": np.arange(10, dtype=float),
            "maintenance_flag": np.ones(10, dtype=int),
        },
        index=idx,
    )
    step = MaintenanceRemovalStep(sample_process_config)
    result = step.process(df)
    assert "maintenance_flag" not in result.columns
    assert result["value"].isna().all()


def test_maintenance_removal_partial_mask(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    flag = np.zeros(10, dtype=int)
    flag[3:6] = 1
    df = pd.DataFrame({"value": np.arange(10, dtype=float), "maintenance_flag": flag}, index=idx)
    step = MaintenanceRemovalStep(sample_process_config)
    result = step.process(df)
    assert result["value"].iloc[3:6].isna().all()
    assert not result["value"].iloc[:3].isna().any()
    assert not result["value"].iloc[6:].isna().any()


def test_maintenance_removal_missing_column_warns(sample_process_config):
    logger = MagicMock()
    idx = pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame({"value": np.arange(5, dtype=float)}, index=idx)
    step = MaintenanceRemovalStep(sample_process_config, logger=logger)
    result = step.process(df)
    pd.testing.assert_frame_equal(result, df)
    assert logger.warning.called


# ----- ImputationStep --------------------------------------------------------


def test_imputation_step_linear_fills_nans(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
    values = np.arange(20, dtype=float)
    values[5] = np.nan
    values[10] = np.nan
    df = pd.DataFrame({"value": values}, index=idx)
    step = ImputationStep(sample_process_config)
    result = step.process(df)
    assert result["value"].isna().sum() == 0


def test_imputation_step_adds_weight_column(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
    values = np.arange(20, dtype=float)
    values[5] = np.nan
    df = pd.DataFrame({"value": values}, index=idx)
    step = ImputationStep(sample_process_config)
    result = step.process(df)
    assert "value__weight" in result.columns
    assert result["value__weight"].iloc[5] == 0
    assert result["value__weight"].iloc[0] == 1


def test_imputation_step_no_weights_when_disabled(sample_process_config):
    cfg = {**sample_process_config}
    cfg["process"] = {**sample_process_config["process"]}
    cfg["process"]["imputation"] = {
        **sample_process_config["process"]["imputation"],
        "add_weights": False,
    }
    idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
    values = np.arange(20, dtype=float)
    values[5] = np.nan
    df = pd.DataFrame({"value": values}, index=idx)
    step = ImputationStep(cfg)
    result = step.process(df)
    assert "value__weight" not in result.columns


def test_imputation_step_psm_branch(sample_process_config):
    cfg = {**sample_process_config}
    cfg["process"] = {**sample_process_config["process"]}
    cfg["process"]["imputation"] = {
        **sample_process_config["process"]["imputation"],
        "method": "psm",
    }
    idx = pd.date_range("2025-01-01", periods=50, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    values = np.sin(np.arange(50) * 0.3) * 10 + 50 + rng.standard_normal(50) * 0.01
    values[20] = np.nan  # single NaN, will be filled by fill_missing_with_mean
    df = pd.DataFrame({"value": values}, index=idx)
    step = ImputationStep(cfg)
    result = step.process(df)
    # PSM/mean branch must have filled the single NaN
    assert not pd.isna(result["value"].iloc[20])


def test_imputation_step_skips_weight_columns_as_input(sample_process_config):
    # Weight columns (suffix '__weight') should be skipped (not re-imputed) by the imputer loop.
    # They may be overwritten by add_weights=True logic, but the imputer should not crash on them.
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "value": np.arange(10, dtype=float),
            "value__weight": [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        index=idx,
    )
    step = ImputationStep(sample_process_config)
    # Smoke test: must not raise even though a weight column already exists
    result = step.process(df)
    assert "value__weight" in result.columns
    assert "value" in result.columns
    # Value column has no NaNs
    assert result["value"].isna().sum() == 0


# ----- MedianFilterStep ------------------------------------------------------


def test_median_filter_applies_to_flow_column(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
    values = np.full(20, 5.0)
    values[10] = 100.0  # spike that median filter (kernel=3) should remove
    df = pd.DataFrame({"flow_primary": values}, index=idx)
    step = MedianFilterStep(sample_process_config)
    result = step.process(df)
    # The spike value at idx 10 should now be 5.0 (median of [5,100,5])
    assert result["flow_primary"].iloc[10] == 5.0


def test_median_filter_skips_non_flow_columns(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
    values = np.full(20, 5.0)
    values[10] = 100.0
    df = pd.DataFrame({"value": values}, index=idx)
    step = MedianFilterStep(sample_process_config)
    result = step.process(df)
    # The 'value' column does not match the 'flow' pattern → spike survives
    assert result["value"].iloc[10] == 100.0


def test_median_filter_skips_weight_columns(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
    flow_values = np.full(20, 5.0)
    flow_values[10] = 100.0
    weight_values = np.ones(20, dtype=int)
    weight_values[10] = 0
    df = pd.DataFrame(
        {"flow_primary": flow_values, "flow_primary__weight": weight_values},
        index=idx,
    )
    step = MedianFilterStep(sample_process_config)
    result = step.process(df)
    # Weight column untouched
    assert result["flow_primary__weight"].iloc[10] == 0


# ----- TemperatureAggregationStep -------------------------------------------


def test_temperature_aggregation_creates_temperature_column(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "channel_0_temperature_1": np.full(10, 10.0),
            "channel_0_temperature_2": np.full(10, 20.0),
            "other": np.arange(10, dtype=float),
        },
        index=idx,
    )
    step = TemperatureAggregationStep(sample_process_config)
    result = step.process(df)
    assert "temperature" in result.columns
    # Mean of 10 and 20 = 15
    assert (result["temperature"] == 15.0).all()
    # Original temperature columns dropped
    assert "channel_0_temperature_1" not in result.columns
    assert "channel_0_temperature_2" not in result.columns
    # Non-temperature column preserved
    assert "other" in result.columns


def test_temperature_aggregation_no_op_when_no_matching_columns(sample_process_config):
    logger = MagicMock()
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({"value": np.arange(10, dtype=float)}, index=idx)
    step = TemperatureAggregationStep(sample_process_config, logger=logger)
    result = step.process(df)
    assert "temperature" not in result.columns
    pd.testing.assert_frame_equal(result, df)
    assert logger.warning.called


def test_temperature_aggregation_ignores_weight_columns(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "channel_0_temperature_1": np.full(10, 10.0),
            "channel_0_temperature_1__weight": np.ones(10, dtype=int),
        },
        index=idx,
    )
    step = TemperatureAggregationStep(sample_process_config)
    result = step.process(df)
    # Temperature aggregated from the single temperature column (not its weight)
    assert "temperature" in result.columns
    assert (result["temperature"] == 10.0).all()
    # Weight column still present, untouched
    assert "channel_0_temperature_1__weight" in result.columns


# ----- WeatherAdjustmentStep ------------------------------------------------


def test_weather_adjustment_no_temperature_col_drops_baseline(sample_process_config):
    # No temperature to adjust, but the baseline is an intermediate and must be
    # dropped so it never reaches training/detection.
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"value": np.arange(10, dtype=float), "exogenous_weather_temperature_baseline": np.full(10, 5.0)},
        index=idx,
    )
    step = WeatherAdjustmentStep(sample_process_config)
    result = step.process(df)
    assert "exogenous_weather_temperature_baseline" not in result.columns
    assert (result["value"] == np.arange(10, dtype=float)).all()


def test_weather_adjustment_no_baseline_col_passes_through(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({"temperature": np.full(10, 20.0)}, index=idx)
    step = WeatherAdjustmentStep(sample_process_config)
    result = step.process(df)
    pd.testing.assert_frame_equal(result, df)


def test_weather_adjustment_subtracts_then_drops_baseline(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"temperature": np.full(10, 20.0), "exogenous_weather_temperature_baseline": np.full(10, 5.0)},
        index=idx,
    )
    step = WeatherAdjustmentStep(sample_process_config)
    result = step.process(df)
    assert (result["temperature"] == 15.0).all()
    # baseline is an intermediate, dropped once it has served the detrend
    assert "exogenous_weather_temperature_baseline" not in result.columns


def test_weather_adjustment_does_not_mutate_input(sample_process_config):
    idx = pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"temperature": np.full(5, 20.0), "exogenous_weather_temperature_baseline": np.full(5, 5.0)},
        index=idx,
    )
    original = df.copy()
    WeatherAdjustmentStep(sample_process_config).process(df)
    pd.testing.assert_frame_equal(df, original)
