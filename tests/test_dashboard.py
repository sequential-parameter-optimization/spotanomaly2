# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for LiveMonitor: run_once() orchestration and the freshness/NaN-tail block.

These tests heavily mock out the domain-layer classes (fetchers, processors,
detectors, report generator) — the goal is to characterise the orchestration
glue and the freshness/NaN-tail block that mutates ``fetch_status['primary']``.
Pinning real domain code requires real models on disk, which is out of scope for
unit tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.dashboard import LiveMonitor
from spotanomaly2.dashboard import live_monitor as live_monitor_module
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
    """tmp workspace where the readiness gate (data + model) returns True."""
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
    sample_config["exogenous"] = []
    sample_config["process"] = {
        **sample_config.get("process", {}),
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
# _check_current_anomalies
# ---------------------------------------------------------------------------


class TestCheckCurrentAnomalies:
    def test_logs_warning_when_anomalies_present(self, sample_config):
        monitor = LiveMonitor(sample_config)
        monitor.logger = MagicMock()

        idx = pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC")
        scores_df = pd.DataFrame({"sensor_a": [0.1, 0.2, 0.95]}, index=idx)
        flags_df = pd.DataFrame({"sensor_a": [0, 0, 1]}, index=idx)
        forecast_df = pd.DataFrame({"sensor_a": [1.0, 1.0, 1.0]}, index=idx)
        results = {"1": (scores_df, flags_df, forecast_df, None, None)}

        monitor._check_current_anomalies(results)

        # At least one warning call mentions ANOMALY DETECTED.
        warning_calls = [str(c) for c in monitor.logger.warning.call_args_list]
        assert any("ANOMALY DETECTED" in s for s in warning_calls)
        assert any("sensor_a" in s for s in warning_calls)
        # Score formatted with 4 decimals.
        assert any("0.9500" in s for s in warning_calls)

    def test_logs_all_clear_when_no_anomalies(self, sample_config):
        monitor = LiveMonitor(sample_config)
        monitor.logger = MagicMock()

        idx = pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC")
        scores_df = pd.DataFrame({"sensor_a": [0.1, 0.2, 0.3]}, index=idx)
        flags_df = pd.DataFrame({"sensor_a": [0, 0, 0]}, index=idx)
        forecast_df = pd.DataFrame({"sensor_a": [1.0, 1.0, 1.0]}, index=idx)
        results = {"1": (scores_df, flags_df, forecast_df, None, None)}

        monitor._check_current_anomalies(results)

        # No warning calls, an "All clear" info call.
        monitor.logger.warning.assert_not_called()
        info_calls = [str(c) for c in monitor.logger.info.call_args_list]
        assert any("All clear" in s for s in info_calls)

    def test_skips_empty_flags(self, sample_config):
        monitor = LiveMonitor(sample_config)
        monitor.logger = MagicMock()

        empty = pd.DataFrame()
        results = {"1": (empty, empty, empty, None, None)}

        # Should not raise even with empty flags.
        monitor._check_current_anomalies(results)


# ---------------------------------------------------------------------------
# run_once() — prerequisite check
# ---------------------------------------------------------------------------


class TestLivePrerequisites:
    def test_run_once_raises_runtime_error_when_not_ready(self, sample_config, tmp_path):
        # No processed data exists → readiness gate is False.
        sample_config["paths"]["processed_dir"] = str(tmp_path / "processed")
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        monitor = LiveMonitor(sample_config)

        with pytest.raises(RuntimeError, match="PREREQUISITE CHECK FAILED"):
            monitor.run_once()


# ---------------------------------------------------------------------------
# run_once() — freshness / NaN-tail block
# ---------------------------------------------------------------------------


class TestLiveFreshnessBlock:
    """Pin the behaviour of the freshness/NaN-tail block.

    The block walks `complete_processed_data` and builds a panel_nan_gaps
    dict that gets attached to fetch_status['primary']. It also downgrades
    primary status to 'degraded' when any panel has either:
      - data older than 1 hour, OR
      - a trailing-NaN tail of 12+ rows.
    """

    def test_empty_df_gives_null_nan_gap_entry(self, ready_workspace):
        """An empty panel df should produce a panel_nan_gaps entry full of None/0.

        NOTE: run_once() falls back to loading the baseline parquet when the live
        df has <MIN_ROWS_FOR_DETECTION rows. To trigger the empty-df branch
        of the freshness block, we must (a) load_live returns the empty df,
        AND (b) the baseline parquet on disk is *also* below the threshold,
        so the fallback doesn't kick in.
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

        monitor = LiveMonitor(config)

        with (
            patch("spotanomaly2.dashboard.live_monitor.resolve_primary_fetcher") as fetcher_cls,
            patch("spotanomaly2.dashboard.live_monitor.DataProcessor") as proc_cls,
            patch("spotanomaly2.dashboard.live_monitor.AnomalyDetector") as det_cls,
            patch.object(monitor._data_manager, "save_processed_data_live"),
            patch.object(monitor._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                monitor._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.dashboard.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det = MagicMock()
            det.run.return_value = {}
            det_cls.return_value = det
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            # Readiness gate needs to still pass — fresh baseline + fresh model.
            # Make the small baseline df recent enough by re-anchoring it.
            now = pd.Timestamp.now(tz="UTC").floor("min")
            recent_small = pd.DataFrame(
                {"sensor_a": np.arange(5, dtype=float)},
                index=pd.date_range(end=now, periods=5, freq="5min", tz="UTC"),
            )
            recent_small.index.name = "timestamp"
            recent_small.to_parquet(paths["processed"] / "panel_1.parquet")

            monitor.run_once()

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

        monitor = LiveMonitor(config)

        with (
            patch("spotanomaly2.dashboard.live_monitor.resolve_primary_fetcher") as fetcher_cls,
            patch("spotanomaly2.dashboard.live_monitor.DataProcessor") as proc_cls,
            patch("spotanomaly2.dashboard.live_monitor.AnomalyDetector") as det_cls,
            patch.object(monitor._data_manager, "save_processed_data_live"),
            patch.object(monitor._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                monitor._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.dashboard.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            monitor.run_once()

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

        monitor = LiveMonitor(config)

        with (
            patch("spotanomaly2.dashboard.live_monitor.resolve_primary_fetcher") as fetcher_cls,
            patch("spotanomaly2.dashboard.live_monitor.DataProcessor") as proc_cls,
            patch("spotanomaly2.dashboard.live_monitor.AnomalyDetector") as det_cls,
            patch.object(monitor._data_manager, "save_processed_data_live"),
            patch.object(monitor._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                monitor._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.dashboard.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            monitor.run_once()

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

        monitor = LiveMonitor(config)

        with (
            patch("spotanomaly2.dashboard.live_monitor.resolve_primary_fetcher") as fetcher_cls,
            patch("spotanomaly2.dashboard.live_monitor.DataProcessor") as proc_cls,
            patch("spotanomaly2.dashboard.live_monitor.AnomalyDetector") as det_cls,
            patch.object(monitor._data_manager, "save_processed_data_live"),
            patch.object(monitor._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                monitor._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.dashboard.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            monitor.run_once()

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

        monitor = LiveMonitor(config)

        with (
            patch("spotanomaly2.dashboard.live_monitor.resolve_primary_fetcher") as fetcher_cls,
            patch("spotanomaly2.dashboard.live_monitor.DataProcessor") as proc_cls,
            patch("spotanomaly2.dashboard.live_monitor.AnomalyDetector") as det_cls,
            patch.object(monitor._data_manager, "save_processed_data_live"),
            patch.object(monitor._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                monitor._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.dashboard.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            monitor.run_once()

            fetch_status = report.generate_report.call_args.kwargs["fetch_status"]
            entry = fetch_status["primary"]["panel_nan_gaps"]["1"]
            # ~2 hours of age.
            assert entry["data_age_seconds"] is not None
            assert entry["data_age_seconds"] > 3600
            assert fetch_status["primary"]["status"] == "degraded"
            assert "stale" in (fetch_status["primary"]["error"] or "").lower()

    def test_nan_only_incremental_data_marks_primary_degraded(self, ready_workspace):
        """When the primary fetcher's run returns NaN-only data, primary status is degraded from the start."""
        config, paths = ready_workspace
        config["report"] = {"enabled": True}
        # Fresh complete data so freshness logic doesn't overwrite us.
        now = pd.Timestamp.now(tz="UTC").floor("min")
        idx = pd.date_range(end=now, periods=300, freq="5min", tz="UTC")
        df = pd.DataFrame({"sensor_a": np.arange(300, dtype=float)}, index=idx)
        df.index.name = "timestamp"
        complete = {"1": df}

        monitor = LiveMonitor(config)

        # Primary returns an empty df.
        empty_idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        nan_only = pd.DataFrame(index=empty_idx)

        with (
            patch("spotanomaly2.dashboard.live_monitor.resolve_primary_fetcher") as fetcher_cls,
            patch("spotanomaly2.dashboard.live_monitor.DataProcessor") as proc_cls,
            patch("spotanomaly2.dashboard.live_monitor.AnomalyDetector") as det_cls,
            patch.object(monitor._data_manager, "save_processed_data_live"),
            patch.object(monitor._data_manager, "load_processed_data_live", return_value=complete),
            patch.object(
                monitor._data_manager,
                "save_detection_results",
                return_value=Path(paths["results"]) / "live",
            ),
            patch("spotanomaly2.dashboard.report_generator.LiveReportGenerator") as report_cls,
        ):
            fetcher_cls.return_value.run.return_value = {"1": nan_only}
            proc_cls.return_value.run.return_value = {"1": _df("2025-01-01", 5)}
            det_cls.return_value.run.return_value = {}
            report = MagicMock()
            report.generate_report.return_value = Path("/tmp/r.html")
            report_cls.return_value = report

            monitor.run_once()

            fetch_status = report.generate_report.call_args.kwargs["fetch_status"]
            assert fetch_status["primary"]["status"] == "degraded"
            assert "NaN" in (fetch_status["primary"]["error"] or "")


# ---------------------------------------------------------------------------
# run_monitoring() — interval loop
# ---------------------------------------------------------------------------


class TestRunMonitoring:
    def test_rejects_interval_lt_1(self, sample_config):
        """Sanity guard: interval must be at least 1."""
        monitor = LiveMonitor(sample_config)
        monitor.logger = MagicMock()
        rc = monitor.run_monitoring(0)
        assert rc != 0
        monitor.logger.error.assert_called()

    def test_loops_until_signal(self, sample_config, monkeypatch):
        """run_monitoring should call run_once() once and stop cleanly on interrupt.

        We patch signal.signal and time.sleep to make the loop deterministic and
        fast, force time.time() far into the future to skip the wait, and raise
        KeyboardInterrupt from run_once() to break the outer loop.
        """
        sample_config["report"] = {"enabled": False}
        monitor = LiveMonitor(sample_config)
        monitor.logger = MagicMock()

        # Don't actually register signal handlers or sleep.
        monkeypatch.setattr(live_monitor_module.signal, "signal", MagicMock())
        monkeypatch.setattr(live_monitor_module.time, "sleep", MagicMock())

        real_time = live_monitor_module.time.time
        call_count = {"n": 0}

        def fake_time():
            call_count["n"] += 1
            return real_time() + 10_000 * call_count["n"]

        monkeypatch.setattr(live_monitor_module.time, "time", fake_time)

        # Raise KeyboardInterrupt from run_once so the outer loop breaks.
        monitor.run_once = MagicMock(side_effect=KeyboardInterrupt())

        rc = monitor.run_monitoring(1)

        assert rc == 0
        assert monitor.run_once.call_count == 1
