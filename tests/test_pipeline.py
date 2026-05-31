# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Pipeline orchestration: per-stage delegation and return values.

These tests heavily mock out the domain-layer classes (fetchers, processors,
trainers, detectors) — the goal is to characterise the orchestration glue and
the artifacts each stage returns. Pinning real domain code requires real models
on disk, which is out of scope for unit tests.

Live-mode behaviour (incremental fetch, freshness/NaN-tail handling, current
anomaly checks) lives in ``LiveMonitor`` and is covered by ``test_dashboard.py``.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from spotanomaly2.application.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(start, periods, freq="5min", cols=("sensor_a",), value=1.0):
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    data = {c: np.full(periods, value, dtype=float) for c in cols}
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Per-stage delegation + return values
# ---------------------------------------------------------------------------


class TestPerStageDelegation:
    def test_download_calls_primary_fetcher_and_returns_raw(self, sample_config):
        pipeline = Pipeline(sample_config)
        with (
            patch("spotanomaly2.application.pipeline.resolve_primary_fetcher") as mock_resolve,
            patch.object(pipeline._data_manager, "save_raw_data") as mock_save,
            patch.object(pipeline._exogenous_download_manager, "download_all") as mock_dl,
        ):
            raw = {"1": _df("2025-01-01", 5)}
            mock_fetcher = MagicMock()
            mock_fetcher.run.return_value = raw
            mock_resolve.return_value = mock_fetcher

            result = pipeline.download()

            mock_resolve.assert_called_once_with(sample_config, pipeline.logger)
            mock_fetcher.run.assert_called_once_with(ignore_cache=False)
            mock_save.assert_called_once()
            mock_dl.assert_called_once()
            assert result is raw

    def test_download_ignore_cache_threads_flag(self, sample_config):
        pipeline = Pipeline(sample_config)
        with (
            patch("spotanomaly2.application.pipeline.resolve_primary_fetcher") as mock_resolve,
            patch.object(pipeline._data_manager, "save_raw_data"),
            patch.object(pipeline._exogenous_download_manager, "download_all") as mock_dl,
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.run.return_value = {"1": _df("2025-01-01", 5)}
            mock_resolve.return_value = mock_fetcher

            pipeline.download(ignore_cache=True)

            mock_fetcher.run.assert_called_once_with(ignore_cache=True)
            assert mock_dl.call_args.kwargs["ignore_cache"] is True

    def test_process_calls_data_processor_after_exogenous_joiner_and_returns_processed(self, sample_config):
        pipeline = Pipeline(sample_config)
        with (
            patch.object(pipeline._data_manager, "load_raw_data") as mock_load,
            patch.object(pipeline._data_manager, "save_processed_data") as mock_save,
            patch.object(pipeline._exogenous_join_manager, "join_all") as mock_join,
            patch("spotanomaly2.application.pipeline.DataProcessor") as mock_proc_cls,
        ):
            raw = {"1": _df("2025-01-01", 5)}
            joined = {"1": _df("2025-01-01", 5, value=7.0)}
            processed = {"1": _df("2025-01-01", 5, value=9.0)}
            mock_load.return_value = raw
            mock_join.return_value = joined
            mock_proc = MagicMock()
            mock_proc.run.return_value = processed
            mock_proc_cls.return_value = mock_proc

            result = pipeline.process()

            mock_load.assert_called_once()
            mock_join.assert_called_once_with(raw)
            mock_proc_cls.assert_called_once_with(sample_config, pipeline.logger)
            # DataProcessor.run receives the joined dict, not the raw dict.
            mock_proc.run.assert_called_once_with(joined)
            mock_save.assert_called_once_with(processed)
            assert result is processed

    def test_train_iterates_panels_and_returns_eval_map(self, sample_config, tmp_path):
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        sample_config["train"]["fallback_model"] = "LightGBM"
        pipeline = Pipeline(sample_config)

        with (
            patch.object(pipeline._data_manager, "load_processed_data") as mock_load,
            patch("spotanomaly2.application.pipeline.ModelTrainer") as mock_trainer_cls,
        ):
            mock_load.return_value = {
                "1": _df("2025-01-01", 5),
                "2": _df("2025-01-02", 5),
            }
            trainer = MagicMock()
            # ModelTrainer.train_panel returns (eval_df, timestamp).
            eval_df = pd.DataFrame({"metric": [0.1]})
            trainer.train_panel.return_value = (eval_df, "20250101_010101")
            mock_trainer_cls.return_value = trainer

            result = pipeline.train()

            assert trainer.train_panel.call_count == 2
            assert set(result) == {"1", "2"}
            assert result["1"][1] == "20250101_010101"
            assert result["1"][0] is eval_df

    def test_detect_delegates_to_anomaly_detector_and_returns_results(self, sample_config):
        pipeline = Pipeline(sample_config)
        with (
            patch.object(pipeline._data_manager, "load_processed_data") as mock_load,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as mock_det_cls,
            patch.object(pipeline._data_manager, "save_detection_results") as mock_save,
        ):
            mock_load.return_value = {"1": _df("2025-01-01", 5)}
            det = MagicMock()
            results = {"1": ("scores", "flags", "forecast", None, None)}
            det.run.return_value = results
            mock_det_cls.return_value = det

            result = pipeline.detect()

            mock_det_cls.assert_called_once_with(sample_config, pipeline.logger)
            det.run.assert_called_once()
            mock_save.assert_called_once()
            assert result is results
