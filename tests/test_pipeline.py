# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Pipeline orchestration: live(), run_all(), per-stage stubs.

These tests heavily mock out the domain-layer classes (fetchers, processors,
trainers, detectors, report generator) — the goal is to characterise the
orchestration glue and the freshness/NaN-tail block that mutates
``fetch_status['primary']``. Pinning real domain code requires real models
on disk, which is out of scope for unit tests.

The freshness block at pipeline.py:479 is exercised via
``TestLiveFreshnessBlock`` (no anomalies, stale data, NaN tails, empty df).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.application.pipeline import Pipeline
from spotanomaly2.infrastructure import storage

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _df(start, periods, freq="5min", cols=("sensor_a",), value=1.0):
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    data = {c: np.full(periods, value, dtype=float) for c in cols}
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def ready_workspace(tmp_path, sample_config):
    """tmp workspace where Pipeline.is_ready() returns True."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    raw_dir = tmp_path / "raw"
    results_dir = tmp_path / "results"
    for d in (processed_dir, models_dir, raw_dir, results_dir):
        d.mkdir()

    sample_config["paths"]["raw_dir"] = str(raw_dir)
    sample_config["paths"]["processed_dir"] = str(processed_dir)
    sample_config["paths"]["models_dir"] = str(models_dir)
    sample_config["paths"]["results_dir"] = str(results_dir)
    sample_config["panels"]["panel_ids"] = ["1"]
    sample_config["train"]["fallback_model"] = "LightGBM"
    sample_config["report"] = {"enabled": False}
    sample_config["exogenous"] = {"enabled": False, "display_name": "Exogenous"}
    sample_config["process"] = {
        **sample_config.get("process", {}),
        "weather": {"enabled": False},
        "imputation": {"weight_suffix": "__weight"},
        "resample": {"freq": "5min"},
    }

    # Fresh panel parquet (with enough rows to clear MIN_ROWS_FOR_DETECTION=200).
    now = pd.Timestamp.now(tz="UTC").floor("min")
    idx = pd.date_range(end=now, periods=250, freq="5min", tz="UTC")
    df = pd.DataFrame({"sensor_a": np.arange(250, dtype=float)}, index=idx)
    df.index.name = "timestamp"
    storage.save_panel_parquet(df, processed_dir, "1")

    model_dir = models_dir / "20251231_120000"
    model_dir.mkdir()
    (model_dir / "fc_model_panel_1.pkl").write_bytes(b"\x00")

    return sample_config, {
        "raw": raw_dir,
        "processed": processed_dir,
        "models": models_dir,
        "results": results_dir,
    }


# ---------------------------------------------------------------------------
# Per-stage delegation
# ---------------------------------------------------------------------------


class TestPerStageDelegation:
    def test_download_calls_primary_fetcher_and_save(self, sample_config):
        pipeline = Pipeline(sample_config)
        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as mock_fetcher_cls,
            patch.object(pipeline._data_manager, "save_raw_data") as mock_save,
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.run.return_value = {"1": _df("2025-01-01", 5)}
            mock_fetcher_cls.return_value = mock_fetcher

            pipeline.download()

            mock_fetcher_cls.assert_called_once_with(sample_config, pipeline.logger)
            mock_fetcher.run.assert_called_once()
            mock_save.assert_called_once()

    def test_process_calls_data_processor_after_merge_exogenous(self, sample_config):
        pipeline = Pipeline(sample_config)
        with (
            patch.object(pipeline._data_manager, "load_raw_data") as mock_load,
            patch.object(pipeline._data_manager, "save_processed_data") as mock_save,
            patch("spotanomaly2.application.pipeline.DataProcessor") as mock_proc_cls,
        ):
            raw = {"1": _df("2025-01-01", 5)}
            processed = {"1": _df("2025-01-01", 5, value=9.0)}
            mock_load.return_value = raw
            mock_proc = MagicMock()
            mock_proc.run.return_value = processed
            mock_proc_cls.return_value = mock_proc

            pipeline.process()

            mock_load.assert_called_once()
            mock_proc_cls.assert_called_once_with(sample_config, pipeline.logger)
            mock_proc.run.assert_called_once()
            mock_save.assert_called_once_with(processed)

    def test_train_iterates_panels_and_delegates_to_trainer(self, sample_config, tmp_path):
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

            # Need the timestamped models dir to exist for eval_df.to_csv.
            (tmp_path / "models" / "20250101_010101").mkdir(parents=True)

            pipeline.train()

            assert trainer.train_panel.call_count == 2

    def test_detect_delegates_to_anomaly_detector(self, sample_config):
        pipeline = Pipeline(sample_config)
        with (
            patch.object(pipeline._data_manager, "load_processed_data") as mock_load,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as mock_det_cls,
            patch.object(pipeline._data_manager, "save_detection_results") as mock_save,
        ):
            mock_load.return_value = {"1": _df("2025-01-01", 5)}
            det = MagicMock()
            det.run.return_value = {}
            mock_det_cls.return_value = det

            pipeline.detect()

            mock_det_cls.assert_called_once_with(sample_config, pipeline.logger)
            det.run.assert_called_once()
            mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# run_all() flag handling
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_predict_only_skips_download_process_train(self, sample_config):
        pipeline = Pipeline(sample_config)
        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.ModelTrainer") as trainer_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "load_processed_data") as mock_load,
            patch.object(pipeline._data_manager, "save_detection_results"),
        ):
            mock_load.return_value = {"1": _df("2025-01-01", 5)}
            det = MagicMock()
            det.run.return_value = {}
            det_cls.return_value = det

            pipeline.run_all(predict_only=True)

            fetcher_cls.assert_not_called()
            proc_cls.assert_not_called()
            trainer_cls.assert_not_called()
            det_cls.assert_called_once()

    def test_skip_download_uses_load_raw_data(self, sample_config, tmp_path):
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        sample_config["train"]["fallback_model"] = "LightGBM"
        pipeline = Pipeline(sample_config)
        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.ModelTrainer") as trainer_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "load_raw_data") as mock_load_raw,
            patch.object(pipeline._data_manager, "save_processed_data"),
            patch.object(pipeline._data_manager, "save_detection_results"),
        ):
            mock_load_raw.return_value = {"1": _df("2025-01-01", 5)}
            proc = MagicMock()
            proc.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value = proc

            trainer = MagicMock()
            trainer.train_panel.return_value = (pd.DataFrame({"m": [0.0]}), "20250101_010101")
            trainer_cls.return_value = trainer
            (tmp_path / "models" / "20250101_010101").mkdir(parents=True)

            det = MagicMock()
            det.run.return_value = {}
            det_cls.return_value = det

            pipeline.run_all(skip_download=True)

            fetcher_cls.assert_not_called()
            mock_load_raw.assert_called_once()

    def test_full_run_invokes_fetcher_then_processor_then_trainer_then_detector(self, sample_config, tmp_path):
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        sample_config["train"]["fallback_model"] = "LightGBM"
        pipeline = Pipeline(sample_config)
        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.ModelTrainer") as trainer_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "save_raw_data"),
            patch.object(pipeline._data_manager, "save_processed_data"),
            patch.object(pipeline._data_manager, "save_detection_results"),
        ):
            fetcher = MagicMock()
            fetcher.run.return_value = {"1": _df("2025-01-01", 5)}
            fetcher_cls.return_value = fetcher

            proc = MagicMock()
            proc.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value = proc

            trainer = MagicMock()
            trainer.train_panel.return_value = (
                pd.DataFrame({"m": [0.0]}),
                "20250101_010101",
            )
            trainer_cls.return_value = trainer
            (tmp_path / "models" / "20250101_010101").mkdir(parents=True)

            det = MagicMock()
            det.run.return_value = {}
            det_cls.return_value = det

            pipeline.run_all()

            fetcher.run.assert_called_once()
            proc.run.assert_called_once()
            trainer.train_panel.assert_called_once()
            det.run.assert_called_once()


# ---------------------------------------------------------------------------
# _check_current_anomalies
# ---------------------------------------------------------------------------


class TestCheckCurrentAnomalies:
    def test_logs_warning_when_anomalies_present(self, sample_config):
        pipeline = Pipeline(sample_config)
        pipeline.logger = MagicMock()

        idx = pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC")
        scores_df = pd.DataFrame({"sensor_a": [0.1, 0.2, 0.95]}, index=idx)
        flags_df = pd.DataFrame({"sensor_a": [0, 0, 1]}, index=idx)
        forecast_df = pd.DataFrame({"sensor_a": [1.0, 1.0, 1.0]}, index=idx)
        results = {"1": (scores_df, flags_df, forecast_df, None, None)}

        pipeline._check_current_anomalies(results)

        # At least one warning call mentions ANOMALY DETECTED.
        warning_calls = [str(c) for c in pipeline.logger.warning.call_args_list]
        assert any("ANOMALY DETECTED" in s for s in warning_calls)
        assert any("sensor_a" in s for s in warning_calls)
        # Score formatted with 4 decimals.
        assert any("0.9500" in s for s in warning_calls)

    def test_logs_all_clear_when_no_anomalies(self, sample_config):
        pipeline = Pipeline(sample_config)
        pipeline.logger = MagicMock()

        idx = pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC")
        scores_df = pd.DataFrame({"sensor_a": [0.1, 0.2, 0.3]}, index=idx)
        flags_df = pd.DataFrame({"sensor_a": [0, 0, 0]}, index=idx)
        forecast_df = pd.DataFrame({"sensor_a": [1.0, 1.0, 1.0]}, index=idx)
        results = {"1": (scores_df, flags_df, forecast_df, None, None)}

        pipeline._check_current_anomalies(results)

        # No warning calls, an "All clear" info call.
        pipeline.logger.warning.assert_not_called()
        info_calls = [str(c) for c in pipeline.logger.info.call_args_list]
        assert any("All clear" in s for s in info_calls)

    def test_skips_empty_flags(self, sample_config):
        pipeline = Pipeline(sample_config)
        pipeline.logger = MagicMock()

        empty = pd.DataFrame()
        results = {"1": (empty, empty, empty, None, None)}

        # Should not raise even with empty flags.
        pipeline._check_current_anomalies(results)


# ---------------------------------------------------------------------------
# live() — prerequisite check
# ---------------------------------------------------------------------------


class TestLivePrerequisites:
    def test_live_raises_runtime_error_when_not_ready(self, sample_config, tmp_path):
        # No processed data exists → is_ready() is False.
        sample_config["paths"]["processed_dir"] = str(tmp_path / "processed")
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        pipeline = Pipeline(sample_config)

        with pytest.raises(RuntimeError, match="PREREQUISITE CHECK FAILED"):
            pipeline.live()


# ---------------------------------------------------------------------------
# live() — freshness / NaN-tail block at pipeline.py:479
# ---------------------------------------------------------------------------


class TestLiveFreshnessBlock:
    """Pin the behaviour of the freshness/NaN-tail block (pipeline.py:479).

    The block walks `complete_processed_data` and builds a panel_nan_gaps
    dict that gets attached to fetch_status['primary']. It also downgrades
    primary status to 'degraded' when any panel has either:
      - data older than 1 hour, OR
      - a trailing-NaN tail of 12+ rows.

    This block is begging for extraction — the tests below pin its current
    behaviour so a future refactor can be checked against them.
    """

    def _run_live_with_complete_data(self, sample_config, paths, complete_processed_data):
        """Drive Pipeline.live() to the freshness block with a controlled DF.

        Mocks downstream classes (PrimaryDataFetcher, DataProcessor,
        AnomalyDetector) so we can inject a controlled
        `complete_processed_data` via DataManager.load_processed_data_live.
        Returns the value of fetch_status passed to the report generator
        (None when report is disabled — in that case we extract via the
        detector spy).
        """
        pipeline = Pipeline(sample_config)

        # Stub all the I/O & domain calls in the live() pipeline.
        # We need to capture fetch_status, which is internal to live().
        # The cleanest hook is to mock save_detection_results AND
        # capture via the panel_nan_gaps assertion through a side-effect.

        captured = {}

        def fake_detector_run(data):
            # Snapshot the data live() passes to the detector so we can
            # validate the freshness logic chose the right frame.
            captured["data_seen_by_detector"] = data
            return {}

        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "save_processed_data_live"),
            patch.object(pipeline._data_manager, "load_processed_data_live") as mock_load_live,
            patch.object(pipeline._data_manager, "save_detection_results") as mock_save,
        ):
            fetcher = MagicMock()
            # Return non-empty new data so primary status starts at "ok".
            fetcher.run.return_value = {"1": _df("2025-01-01", 5)}
            fetcher_cls.return_value = fetcher

            proc = MagicMock()
            proc.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value = proc

            mock_load_live.return_value = complete_processed_data

            det = MagicMock()
            det.run.side_effect = fake_detector_run
            det_cls.return_value = det

            mock_save.return_value = Path(paths["results"]) / "live"

            results = pipeline.live()

        return results, captured

    def test_empty_df_gives_null_nan_gap_entry(self, ready_workspace):
        """An empty panel df should produce a panel_nan_gaps entry full of None/0.

        NOTE: live() falls back to loading the baseline parquet when the live
        df has <MIN_ROWS_FOR_DETECTION rows. To trigger the empty-df branch
        of the freshness block, we must (a) load_live returns the empty df,
        AND (b) the baseline parquet on disk is *also* below the threshold,
        so the fallback at pipeline.py:459-470 doesn't kick in.
        """
        config, paths = ready_workspace
        config["report"] = {"enabled": True}

        # Overwrite baseline parquet with a tiny df (< MIN_ROWS_FOR_DETECTION).
        small_idx = pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")
        small = pd.DataFrame({"sensor_a": np.arange(5, dtype=float)}, index=small_idx)
        small.index.name = "timestamp"
        small.to_parquet(paths["processed"] / "panel_1.parquet")

        empty_df = pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
        complete = {"1": empty_df}

        pipeline = Pipeline(config)

        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "save_processed_data_live"),
            patch.object(pipeline._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                pipeline._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.domain.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det = MagicMock()
            det.run.return_value = {}
            det_cls.return_value = det
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            # is_ready() needs to still pass — fresh baseline + fresh model.
            # Make the small baseline df recent enough by re-anchoring it.
            now = pd.Timestamp.now(tz="UTC").floor("min")
            recent_small = pd.DataFrame(
                {"sensor_a": np.arange(5, dtype=float)},
                index=pd.date_range(end=now, periods=5, freq="5min", tz="UTC"),
            )
            recent_small.index.name = "timestamp"
            recent_small.to_parquet(paths["processed"] / "panel_1.parquet")

            pipeline.live()

            call_kwargs = report.generate_report.call_args.kwargs
            fetch_status = call_kwargs["fetch_status"]
            assert "primary" in fetch_status
            panel_gaps = fetch_status["primary"]["panel_nan_gaps"]
            assert "1" in panel_gaps
            entry = panel_gaps["1"]
            # Empty-df branch: all entries None / 0.
            assert entry["trailing_nan_rows"] == 0
            assert entry["trailing_nan_duration"] is None
            assert entry["last_valid_timestamp"] is None
            assert entry["latest_timestamp"] is None
            assert entry["data_age_seconds"] is None

    def test_no_trailing_nans_yields_zero_count_and_recent_age(self, ready_workspace):
        config, paths = ready_workspace
        config["report"] = {"enabled": True}
        now = pd.Timestamp.now(tz="UTC").floor("min")
        idx = pd.date_range(end=now, periods=300, freq="5min", tz="UTC")
        # All values present — no trailing NaNs.
        df = pd.DataFrame({"sensor_a": np.arange(300, dtype=float)}, index=idx)
        df.index.name = "timestamp"
        complete = {"1": df}

        pipeline = Pipeline(config)

        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "save_processed_data_live"),
            patch.object(pipeline._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                pipeline._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.domain.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            pipeline.live()

            fetch_status = report.generate_report.call_args.kwargs["fetch_status"]
            entry = fetch_status["primary"]["panel_nan_gaps"]["1"]
            assert entry["trailing_nan_rows"] == 0
            assert entry["trailing_nan_duration"] is None
            assert entry["last_valid_timestamp"] == idx.max().isoformat()
            assert entry["latest_timestamp"] == idx.max().isoformat()
            # Data is recent (<1h old).
            assert entry["data_age_seconds"] is not None
            assert entry["data_age_seconds"] < 3600
            # Primary remains ok.
            assert fetch_status["primary"]["status"] == "ok"

    def test_five_trailing_nan_rows_counted_correctly(self, ready_workspace):
        """A trailing tail of 5 all-NaN sensor rows should report trailing_nan_rows=5."""
        config, paths = ready_workspace
        config["report"] = {"enabled": True}
        now = pd.Timestamp.now(tz="UTC").floor("min")
        idx = pd.date_range(end=now, periods=300, freq="5min", tz="UTC")
        values = np.arange(300, dtype=float)
        values[-5:] = np.nan
        df = pd.DataFrame({"sensor_a": values}, index=idx)
        df.index.name = "timestamp"
        complete = {"1": df}

        pipeline = Pipeline(config)

        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "save_processed_data_live"),
            patch.object(pipeline._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                pipeline._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.domain.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            pipeline.live()

            fetch_status = report.generate_report.call_args.kwargs["fetch_status"]
            entry = fetch_status["primary"]["panel_nan_gaps"]["1"]
            assert entry["trailing_nan_rows"] == 5
            # Duration formatted as str(offset) — for 5min freq this is
            # "<25 * Minutes>", not a Timedelta string. Pin current shape.
            from pandas.tseries.frequencies import to_offset

            assert entry["trailing_nan_duration"] == str(5 * to_offset("5min"))
            # last_valid_timestamp is 5 steps before the latest.
            assert entry["last_valid_timestamp"] == idx[-6].isoformat()
            assert entry["latest_timestamp"] == idx.max().isoformat()
            # 5 NaN rows is below the 12-row threshold, so status stays ok
            # (unless data is also stale, which it isn't in this fixture).
            assert fetch_status["primary"]["status"] == "ok"

    def test_twelve_trailing_nan_rows_downgrades_status_to_degraded(self, ready_workspace):
        """A trailing tail of 12 all-NaN rows must downgrade primary status,
        *without* tripping the stale-data branch (so we can verify the
        NaN-tail branch picks the right error message).
        """
        config, paths = ready_workspace
        config["report"] = {"enabled": True}
        # End the index 30 min in the future so that even after 12*5min=60min
        # of trailing NaNs, last_valid is ~30min ago — fresh, not stale.
        future_end = pd.Timestamp.now(tz="UTC").floor("min") + pd.Timedelta("30min")
        idx = pd.date_range(end=future_end, periods=300, freq="5min", tz="UTC")
        values = np.arange(300, dtype=float)
        values[-12:] = np.nan
        df = pd.DataFrame({"sensor_a": values}, index=idx)
        df.index.name = "timestamp"
        complete = {"1": df}

        pipeline = Pipeline(config)

        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "save_processed_data_live"),
            patch.object(pipeline._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                pipeline._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.domain.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            pipeline.live()

            fetch_status = report.generate_report.call_args.kwargs["fetch_status"]
            entry = fetch_status["primary"]["panel_nan_gaps"]["1"]
            assert entry["trailing_nan_rows"] == 12
            assert fetch_status["primary"]["status"] == "degraded"
            assert "NaN gaps" in (fetch_status["primary"]["error"] or "")

    def test_stale_last_valid_downgrades_status_to_degraded(self, ready_workspace):
        """Data older than 1 hour must downgrade primary status with the stale message."""
        config, paths = ready_workspace
        config["report"] = {"enabled": True}
        # Data ending 2 hours ago.
        end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(hours=2)
        idx = pd.date_range(end=end, periods=300, freq="5min", tz="UTC")
        df = pd.DataFrame({"sensor_a": np.arange(300, dtype=float)}, index=idx)
        df.index.name = "timestamp"
        complete = {"1": df}

        pipeline = Pipeline(config)

        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "save_processed_data_live"),
            patch.object(pipeline._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                pipeline._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.domain.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            pipeline.live()

            fetch_status = report.generate_report.call_args.kwargs["fetch_status"]
            entry = fetch_status["primary"]["panel_nan_gaps"]["1"]
            # ~2 hours of age.
            assert entry["data_age_seconds"] is not None
            assert entry["data_age_seconds"] > 3600
            assert fetch_status["primary"]["status"] == "degraded"
            assert "stale" in (fetch_status["primary"]["error"] or "").lower()

    def test_nan_only_incremental_data_marks_primary_degraded(self, ready_workspace):
        """When PrimaryDataFetcher.run returns NaN-only data, primary status is degraded from the start."""
        config, paths = ready_workspace
        config["report"] = {"enabled": True}
        # Fresh complete data so freshness logic doesn't overwrite us.
        now = pd.Timestamp.now(tz="UTC").floor("min")
        idx = pd.date_range(end=now, periods=300, freq="5min", tz="UTC")
        df = pd.DataFrame({"sensor_a": np.arange(300, dtype=float)}, index=idx)
        df.index.name = "timestamp"
        complete = {"1": df}

        pipeline = Pipeline(config)

        # Primary returns an empty df.
        empty_idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        nan_only = pd.DataFrame(index=empty_idx)

        with (
            patch("spotanomaly2.application.pipeline.PrimaryDataFetcher") as fetcher_cls,
            patch("spotanomaly2.application.pipeline.DataProcessor") as proc_cls,
            patch("spotanomaly2.application.pipeline.AnomalyDetector") as det_cls,
            patch.object(pipeline._data_manager, "save_processed_data_live"),
            patch.object(pipeline._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                pipeline._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.domain.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": nan_only}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            pipeline.live()

            fetch_status = report.generate_report.call_args.kwargs["fetch_status"]
            assert fetch_status["primary"]["status"] == "degraded"
            assert "NaN" in (fetch_status["primary"]["error"] or "")
