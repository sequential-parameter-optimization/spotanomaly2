# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for spotanomaly2.domain.data_processor.DataProcessor orchestration."""

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain.data_processor import DataProcessor
from spotanomaly2.domain.processing_steps import (
    ImputationStep,
    MaintenanceRemovalStep,
    ManualOutlierRemovalStep,
    MedianFilterStep,
    ResampleStep,
    TemperatureAggregationStep,
    WeatherAdjustmentStep,
)


def test_constructs_base_and_post_outlier_steps_in_order(sample_config):
    proc = DataProcessor(sample_config)

    # _base_steps: ResampleStep, MaintenanceRemovalStep
    assert isinstance(proc._base_steps[0], ResampleStep)
    assert isinstance(proc._base_steps[1], MaintenanceRemovalStep)
    assert len(proc._base_steps) == 2

    # _post_outlier_steps: ImputationStep, MedianFilterStep, TemperatureAggregationStep, WeatherAdjustmentStep
    assert isinstance(proc._post_outlier_steps[0], ImputationStep)
    assert isinstance(proc._post_outlier_steps[1], MedianFilterStep)
    assert isinstance(proc._post_outlier_steps[2], TemperatureAggregationStep)
    assert isinstance(proc._post_outlier_steps[3], WeatherAdjustmentStep)
    assert len(proc._post_outlier_steps) == 4


def test_default_logger_used_when_none_given(sample_config):
    proc = DataProcessor(sample_config)
    assert proc.logger is not None
    assert hasattr(proc.logger, "info")


def test_custom_logger_used_when_passed(sample_config):
    custom = logging.getLogger("custom_test_logger")
    proc = DataProcessor(sample_config, logger=custom)
    assert proc.logger is custom


def test_weather_disabled_no_fetcher(sample_config):
    # sample_config has weather.enabled = False
    proc = DataProcessor(sample_config)
    weather_step = proc._post_outlier_steps[3]
    assert weather_step.weather_fetcher is None


def test_weather_enabled_but_missing_coords_logs_warning(sample_config, caplog):
    cfg = {**sample_config}
    cfg["process"] = {**sample_config["process"]}
    cfg["process"]["weather"] = {"enabled": True, "latitude": None, "longitude": None}
    with caplog.at_level(logging.WARNING):
        proc = DataProcessor(cfg)
    weather_step = proc._post_outlier_steps[3]
    assert weather_step.weather_fetcher is None
    assert any("latitude" in rec.message.lower() for rec in caplog.records)


def test_weather_enabled_with_coords_creates_fetcher(sample_config):
    cfg = {**sample_config}
    cfg["process"] = {**sample_config["process"]}
    cfg["process"]["weather"] = {
        "enabled": True,
        "latitude": 52.0,
        "longitude": 13.0,
        "use_forecast": True,
    }
    # Patch WeatherFetcher to avoid real I/O during construction
    with patch("spotanomaly2.domain.data_processor.WeatherFetcher") as mock_fetcher:
        proc = DataProcessor(cfg)
    assert mock_fetcher.called
    weather_step = proc._post_outlier_steps[3]
    assert weather_step.weather_fetcher is not None


def _make_panel_df(n=60, freq="1min"):
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "channel_0_flow_primary": rng.standard_normal(n) + 10.0,
            "channel_0_temperature_1": rng.standard_normal(n) + 15.0,
            "channel_0_maintenance_flag": np.zeros(n, dtype=int),
        },
        index=idx,
    )


def test_process_all_panels_returns_dict_with_preserved_keys(sample_config):
    proc = DataProcessor(sample_config)
    panel_data = {"1": _make_panel_df(), "2": _make_panel_df(n=80, freq="1min")}
    out = proc.process_all_panels(panel_data)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"1", "2"}
    for v in out.values():
        assert isinstance(v, pd.DataFrame)


def test_run_returns_same_as_process_all_panels(sample_config):
    proc = DataProcessor(sample_config)
    panel_data = {"1": _make_panel_df()}
    out_run = proc.run(panel_data)
    # Construct a fresh processor (steps are stateless w.r.t. df) and verify shape
    proc2 = DataProcessor(sample_config)
    out_processed = proc2.process_all_panels({"1": _make_panel_df()})
    assert set(out_run.keys()) == set(out_processed.keys())
    pd.testing.assert_frame_equal(out_run["1"], out_processed["1"])


def test_manual_outlier_removal_step_uses_panel_id(sample_config):
    # Enable manual outliers for panel "1" only
    cfg = {**sample_config}
    cfg["process"] = {**sample_config["process"]}
    cfg["process"]["manual_outliers"] = {
        "enabled": True,
        "panels": {
            "1": {"columns": {"channel_0_flow_primary": {"lower": -1000.0, "upper": 1000.0}}},
        },
    }

    captured_panel_ids = []

    real_process = ManualOutlierRemovalStep.process

    def spy_process(self, df):
        captured_panel_ids.append(self.panel_id)
        return real_process(self, df)

    with patch.object(ManualOutlierRemovalStep, "process", new=spy_process):
        proc = DataProcessor(cfg)
        proc.process_all_panels({"1": _make_panel_df(), "2": _make_panel_df()})

    # ManualOutlierRemovalStep was invoked once per panel, with the correct panel_id
    assert captured_panel_ids == ["1", "2"]


def test_process_panel_passes_panel_id_into_manual_outlier_step(sample_config):
    proc = DataProcessor(sample_config)
    df = _make_panel_df()
    captured = {}

    real_init = ManualOutlierRemovalStep.__init__

    def spy_init(self, config, logger=None, panel_id=None):
        captured["panel_id"] = panel_id
        real_init(self, config, logger, panel_id)

    with patch.object(ManualOutlierRemovalStep, "__init__", new=spy_init):
        proc.process_panel(df, panel_id="42")

    assert captured["panel_id"] == "42"
