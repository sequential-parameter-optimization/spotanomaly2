"""Test Pipeline instantiation and individual step mocking."""

from unittest.mock import MagicMock, patch

import pandas as pd

from spotanomaly2.application.data_manager import DataManager
from spotanomaly2.application.pipeline import Pipeline


def test_pipeline_instantiation(sample_config):
    pipeline = Pipeline(sample_config)
    assert pipeline is not None
    assert pipeline.config == sample_config


def test_pipeline_detect_instantiates_detector(sample_config):
    """Verify detect() creates an AnomalyDetector and calls run()."""
    pipeline = Pipeline(sample_config)

    with (
        patch("spotanomaly2.application.pipeline.AnomalyDetector") as mock_detector,
        patch.object(pipeline._data_manager, "load_processed_data") as mock_load,
        patch.object(pipeline._data_manager, "save_detection_results"),
    ):
        mock_load.return_value = {"1": pd.DataFrame({"a": [1, 2, 3]})}
        mock_instance = MagicMock()
        mock_instance.run.return_value = {}
        mock_detector.return_value = mock_instance

        pipeline.detect()

        mock_detector.assert_called_once_with(sample_config, pipeline.logger)
        mock_instance.run.assert_called_once()


def test_pipeline_train_loads_processed_data(sample_config):
    """Verify train() loads processed data and iterates panels."""
    pipeline = Pipeline(sample_config)

    with patch.object(pipeline._data_manager, "load_processed_data") as mock_load:
        mock_load.return_value = {}

        pipeline.train()

        mock_load.assert_called_once()


def test_save_processed_data_live_refreshes_stale_live_from_baseline(tmp_path, sample_config):
    """Regression: when baseline is far ahead of stale live, refresh from baseline before merging.

    Without this, an incremental fetch run after a fresh `all` run produces a multi-day
    gap between live's last timestamp and the new incremental window.
    """
    config = {
        **sample_config,
        "paths": {**sample_config["paths"], "processed_dir": str(tmp_path / "processed")},
    }
    processed_dir = tmp_path / "processed"
    live_dir = processed_dir / "live"
    processed_dir.mkdir(parents=True)
    live_dir.mkdir()

    baseline_idx = pd.date_range("2025-01-01", "2025-01-10", freq="5min", tz="UTC")
    baseline_df = pd.DataFrame({"sensor_a": 1.0, "sensor_a__weight": 1}, index=baseline_idx)
    baseline_df.index.name = "timestamp"
    baseline_df.to_parquet(processed_dir / "panel_1.parquet")

    stale_idx = pd.date_range("2025-01-01", "2025-01-03", freq="5min", tz="UTC")
    stale_df = pd.DataFrame({"sensor_a": 2.0, "sensor_a__weight": 1}, index=stale_idx)
    stale_df.index.name = "timestamp"
    stale_df.to_parquet(live_dir / "panel_1.parquet")

    new_idx = pd.date_range("2025-01-10 00:05", periods=12, freq="5min", tz="UTC")
    new_df = pd.DataFrame({"sensor_a": 3.0, "sensor_a__weight": 1}, index=new_idx)
    new_df.index.name = "timestamp"

    DataManager(config).save_processed_data_live({"1": new_df})

    saved = pd.read_parquet(live_dir / "panel_1.parquet").sort_index()
    diffs = saved.index.to_series().diff()
    big_gaps = diffs[diffs > pd.Timedelta(hours=1)]
    assert big_gaps.empty, f"Stale live should be re-bootstrapped from baseline; found gaps: {big_gaps.tolist()}"
    assert saved.index.max() >= new_idx.max()
