# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotanomaly2.application.data_manager.DataManager.

Covers parquet round-trips, detection-results layout, and the
five-level-nested save_processed_data_live() bootstrap / refresh / merge
state machine.
"""

import json

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.application.data_manager import DataManager


def _make_df(start: str, periods: int, freq: str = "5min", value: float = 1.0) -> pd.DataFrame:
    """Build a small panel DataFrame with a sensor column + matching weight."""
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "sensor_a": np.full(periods, value, dtype=float),
            "sensor_a__weight": np.ones(periods, dtype=int),
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def dm_workspace(tmp_path, sample_config):
    """Build a DataManager pointed at tmp_path with all four storage dirs configured.

    Returns (config, paths_dict). The directories are *not* created — the
    individual tests exercise the create-on-write behaviour where relevant.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"

    sample_config["paths"]["raw_dir"] = str(raw_dir)
    sample_config["paths"]["processed_dir"] = str(processed_dir)
    sample_config["paths"]["results_dir"] = str(results_dir)
    sample_config["paths"]["models_dir"] = str(models_dir)
    sample_config["panels"]["panel_ids"] = ["1", "2"]
    sample_config["fetch"] = {"start_date": "2025-01-01T00:00:00+00:00"}

    paths = {
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "results_dir": results_dir,
        "models_dir": models_dir,
    }
    return sample_config, paths


# ---------------------------------------------------------------------------
# Processed data: round-trip
# ---------------------------------------------------------------------------


class TestProcessedDataRoundTrip:
    def test_save_then_load_preserves_panels(self, dm_workspace):
        config, paths = dm_workspace
        dm = DataManager(config)

        df1 = _make_df("2025-01-01", periods=10, value=1.5)
        df2 = _make_df("2025-01-02", periods=20, value=2.5)
        dm.save_processed_data({"1": df1, "2": df2})

        loaded = dm.load_processed_data()
        assert set(loaded.keys()) == {"1", "2"}
        # Parquet round-trip drops the DatetimeIndex freq attribute; compare
        # values + index entries without comparing freq.
        pd.testing.assert_frame_equal(
            loaded["1"].sort_index(),
            df1.sort_index(),
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            loaded["2"].sort_index(),
            df2.sort_index(),
            check_freq=False,
        )

    def test_save_processed_creates_directory_on_demand(self, dm_workspace):
        config, paths = dm_workspace
        assert not paths["processed_dir"].exists()
        dm = DataManager(config)
        dm.save_processed_data({"1": _make_df("2025-01-01", periods=5)})
        assert (paths["processed_dir"] / "panel_1.parquet").exists()

    def test_load_processed_missing_panel_raises(self, dm_workspace):
        config, paths = dm_workspace
        paths["processed_dir"].mkdir()
        # Only panel 1 saved, but config asks for both.
        DataManager(config).save_processed_data({"1": _make_df("2025-01-01", periods=5)})
        with pytest.raises(FileNotFoundError):
            DataManager(config).load_processed_data()


# ---------------------------------------------------------------------------
# Raw data: save + metadata
# ---------------------------------------------------------------------------


class TestRawData:
    def test_save_raw_data_versioned_writes_metadata(self, dm_workspace):
        config, paths = dm_workspace
        dm = DataManager(config)
        df1 = _make_df("2025-01-01", periods=10)
        df2 = _make_df("2025-01-02", periods=15)
        dm.save_raw_data({"1": df1, "2": df2})

        version_dirs = [d for d in paths["raw_dir"].iterdir() if d.is_dir()]
        assert len(version_dirs) == 1
        version_dir = version_dirs[0]

        assert (version_dir / "panel_1.parquet").exists()
        assert (version_dir / "panel_2.parquet").exists()
        meta_path = version_dir / "meta.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["panel_ids"] == ["1", "2"]
        assert meta["row_counts"] == {"1": 10, "2": 15}
        assert meta["download_timestamp"] == version_dir.name
        # actual_start/end reflect the union of panel data.
        assert "actual_start" in meta and "actual_end" in meta

    def test_save_raw_data_live_uses_live_subdir(self, dm_workspace):
        config, paths = dm_workspace
        dm = DataManager(config)
        df1 = _make_df("2025-01-01", periods=5)
        dm.save_raw_data({"1": df1}, live=True)

        live_dir = paths["raw_dir"] / "live"
        assert live_dir.is_dir()
        assert (live_dir / "panel_1.parquet").exists()
        meta = json.loads((live_dir / "meta.json").read_text())
        assert "last_update" in meta
        assert "download_timestamp" not in meta


# ---------------------------------------------------------------------------
# Detection results
# ---------------------------------------------------------------------------


class TestSaveDetectionResults:
    def _build_results(self, n: int = 5):
        idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
        scores = pd.DataFrame({"sensor_a": np.linspace(0, 1, n)}, index=idx)
        flags = pd.DataFrame({"sensor_a": np.zeros(n, dtype=int)}, index=idx)
        forecast = pd.DataFrame({"sensor_a": np.linspace(10, 11, n)}, index=idx)
        contributions = pd.DataFrame({"contrib_a": np.linspace(0, 0.1, n)}, index=idx)
        per_channel = {
            "scores": pd.DataFrame({"sensor_a": np.linspace(0, 1, n)}, index=idx),
            "flags": pd.DataFrame({"sensor_a": np.zeros(n, dtype=int)}, index=idx),
            "flags_combined": pd.DataFrame({"sensor_a": np.zeros(n, dtype=int)}, index=idx),
            "thresholds": pd.DataFrame({"sensor_a": np.full(n, 0.99)}, index=idx),
        }
        return scores, flags, forecast, contributions, per_channel

    def test_writes_all_files_in_timestamped_dir(self, dm_workspace):
        config, paths = dm_workspace
        dm = DataManager(config)
        results = {"1": self._build_results()}

        out_dir = dm.save_detection_results(results, timestamp="20250101_120000")

        assert out_dir == paths["results_dir"] / "20250101_120000"
        assert (out_dir / "panel_1_scores.csv").exists()
        assert (out_dir / "panel_1_flags.csv").exists()
        assert (out_dir / "panel_1_forecast.csv").exists()
        assert (out_dir / "panel_1_contributions.parquet").exists()
        assert (out_dir / "panel_1_per_channel_scores.csv").exists()
        assert (out_dir / "panel_1_per_channel_flags.csv").exists()
        assert (out_dir / "panel_1_per_channel_flags_combined.csv").exists()
        assert (out_dir / "panel_1_per_channel_thresholds.csv").exists()

    def test_live_mode_uses_live_subdir(self, dm_workspace):
        config, paths = dm_workspace
        dm = DataManager(config)
        results = {"1": self._build_results()}

        out_dir = dm.save_detection_results(results, live_mode=True)

        assert out_dir == paths["results_dir"] / "live"
        assert (out_dir / "panel_1_scores.csv").exists()

    def test_omits_contributions_when_none(self, dm_workspace):
        config, paths = dm_workspace
        dm = DataManager(config)
        scores, flags, forecast, _contrib, per_ch = self._build_results()
        results = {"1": (scores, flags, forecast, None, per_ch)}

        out_dir = dm.save_detection_results(results, timestamp="20250101_120000")

        assert not (out_dir / "panel_1_contributions.parquet").exists()

    def test_omits_per_channel_when_none(self, dm_workspace):
        config, paths = dm_workspace
        dm = DataManager(config)
        scores, flags, forecast, contrib, _per_ch = self._build_results()
        results = {"1": (scores, flags, forecast, contrib, None)}

        out_dir = dm.save_detection_results(results, timestamp="20250101_120000")

        # Scores still present, per_channel sidecars absent.
        assert (out_dir / "panel_1_scores.csv").exists()
        assert not (out_dir / "panel_1_per_channel_scores.csv").exists()


# ---------------------------------------------------------------------------
# save_processed_data_live — the 215-line nested state machine
# ---------------------------------------------------------------------------


class TestSaveProcessedDataLiveBootstrap:
    """Bootstrap from baseline when no live file exists."""

    def test_bootstrap_from_baseline_when_no_live_file(self, dm_workspace):
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        processed_dir = paths["processed_dir"]
        processed_dir.mkdir()

        # Baseline exists, but no live file yet.
        baseline_idx = pd.date_range("2025-01-01", periods=200, freq="5min", tz="UTC")
        baseline = pd.DataFrame(
            {"sensor_a": np.linspace(1, 2, 200), "sensor_a__weight": 1},
            index=baseline_idx,
        )
        baseline.index.name = "timestamp"
        baseline.to_parquet(processed_dir / "panel_1.parquet")

        # New incremental window contiguous with baseline end.
        new_start = baseline_idx[-1] + pd.Timedelta("5min")
        new_idx = pd.date_range(new_start, periods=10, freq="5min")
        new_df = pd.DataFrame(
            {"sensor_a": np.full(10, 9.0), "sensor_a__weight": 1},
            index=new_idx,
        )
        new_df.index.name = "timestamp"

        DataManager(config).save_processed_data_live({"1": new_df})

        live_path = processed_dir / "live" / "panel_1.parquet"
        assert live_path.exists()
        live = pd.read_parquet(live_path).sort_index()
        # Should contain baseline + new rows.
        assert len(live) == len(baseline) + len(new_df)
        assert live.index.max() == new_idx.max()

    def test_bootstrap_raises_when_no_baseline(self, dm_workspace):
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        # processed_dir does not yet exist — neither baseline nor live.
        new_df = _make_df("2025-01-01", periods=5)

        with pytest.raises(RuntimeError, match="Cannot bootstrap live data"):
            DataManager(config).save_processed_data_live({"1": new_df})

    @pytest.mark.xfail(
        reason=(
            "PRODUCTION BUG: the RuntimeError for 'baseline outdated vs raw' "
            "(data_manager.py:265) is raised inside a try/except Exception "
            "block (data_manager.py:276) that downgrades it to a warning. "
            "The error message is logged but bootstrap proceeds anyway, "
            "silently producing a multi-day gap. This guard never trips."
        )
    )
    def test_bootstrap_raises_when_baseline_older_than_raw_by_more_than_1_day(self, dm_workspace):
        """If raw has data far newer than baseline, bootstrap aborts with documented message."""
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        processed_dir = paths["processed_dir"]
        raw_dir = paths["raw_dir"]
        processed_dir.mkdir()
        raw_dir.mkdir()

        # Baseline from 2025-01-01.
        baseline_idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
        baseline = pd.DataFrame(
            {"sensor_a": 1.0, "sensor_a__weight": 1},
            index=baseline_idx,
        )
        baseline.index.name = "timestamp"
        baseline.to_parquet(processed_dir / "panel_1.parquet")

        # Raw from 2025-02-01 (>1 day gap).
        raw_version = raw_dir / "20250201_000000"
        raw_version.mkdir()
        raw_idx = pd.date_range("2025-02-01", periods=10, freq="5min", tz="UTC")
        raw = pd.DataFrame(
            {"sensor_a": 1.0, "sensor_a__weight": 1},
            index=raw_idx,
        )
        raw.index.name = "timestamp"
        raw.to_parquet(raw_version / "panel_1.parquet")

        new_df = _make_df("2025-02-01", periods=5)

        with pytest.raises(RuntimeError, match="baseline is outdated"):
            DataManager(config).save_processed_data_live({"1": new_df})


class TestSaveProcessedDataLiveRefresh:
    """Refresh paths: schema mismatch & stale live."""

    def test_refresh_from_baseline_when_schema_mismatch(self, dm_workspace):
        """When baseline has new columns vs. live, live should be re-bootstrapped."""
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        processed_dir = paths["processed_dir"]
        live_dir = processed_dir / "live"
        processed_dir.mkdir()
        live_dir.mkdir()

        # Baseline has *two* sensors (recent re-process added a column).
        baseline_idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
        baseline = pd.DataFrame(
            {
                "sensor_a": 1.0,
                "sensor_a__weight": 1,
                "sensor_b": 2.0,
                "sensor_b__weight": 1,
            },
            index=baseline_idx,
        )
        baseline.index.name = "timestamp"
        baseline.to_parquet(processed_dir / "panel_1.parquet")

        # Live still has the old single-sensor schema, but is otherwise fresh
        # (overlaps baseline → not stale).
        live_idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
        live = pd.DataFrame(
            {"sensor_a": 9.0, "sensor_a__weight": 1},
            index=live_idx,
        )
        live.index.name = "timestamp"
        live.to_parquet(live_dir / "panel_1.parquet")

        # New incremental window in the new schema.
        new_idx = pd.date_range(baseline_idx[-1] + pd.Timedelta("5min"), periods=5, freq="5min")
        new_df = pd.DataFrame(
            {
                "sensor_a": 7.0,
                "sensor_a__weight": 1,
                "sensor_b": 8.0,
                "sensor_b__weight": 1,
            },
            index=new_idx,
        )
        new_df.index.name = "timestamp"

        DataManager(config).save_processed_data_live({"1": new_df})

        saved = pd.read_parquet(live_dir / "panel_1.parquet").sort_index()
        # After re-bootstrap from baseline, sensor_b is present everywhere baseline covered.
        assert "sensor_b" in saved.columns
        # Baseline values should dominate where live was diverging.
        assert saved.loc[baseline_idx[0], "sensor_a"] == 1.0

    def test_refresh_from_baseline_when_live_stale(self, dm_workspace):
        """Baseline ends >1h after live → re-bootstrap from baseline before merging."""
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        processed_dir = paths["processed_dir"]
        live_dir = processed_dir / "live"
        processed_dir.mkdir()
        live_dir.mkdir()

        baseline_idx = pd.date_range("2025-01-01", "2025-01-10", freq="5min", tz="UTC")
        baseline = pd.DataFrame(
            {"sensor_a": 1.0, "sensor_a__weight": 1},
            index=baseline_idx,
        )
        baseline.index.name = "timestamp"
        baseline.to_parquet(processed_dir / "panel_1.parquet")

        stale_idx = pd.date_range("2025-01-01", "2025-01-03", freq="5min", tz="UTC")
        stale = pd.DataFrame(
            {"sensor_a": 2.0, "sensor_a__weight": 1},
            index=stale_idx,
        )
        stale.index.name = "timestamp"
        stale.to_parquet(live_dir / "panel_1.parquet")

        new_idx = pd.date_range("2025-01-10 00:05", periods=12, freq="5min", tz="UTC")
        new_df = pd.DataFrame(
            {"sensor_a": 3.0, "sensor_a__weight": 1},
            index=new_idx,
        )
        new_df.index.name = "timestamp"

        DataManager(config).save_processed_data_live({"1": new_df})

        saved = pd.read_parquet(live_dir / "panel_1.parquet").sort_index()
        diffs = saved.index.to_series().diff()
        big_gaps = diffs[diffs > pd.Timedelta(hours=1)]
        assert big_gaps.empty, f"Gaps remain after stale-live refresh: {big_gaps.tolist()}"
        assert saved.index.max() == new_idx.max()


class TestSaveProcessedDataLiveAppend:
    """Normal append: live exists, no schema mismatch, no staleness."""

    def test_append_merges_new_after_existing(self, dm_workspace):
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        processed_dir = paths["processed_dir"]
        live_dir = processed_dir / "live"
        processed_dir.mkdir()
        live_dir.mkdir()

        # Baseline matches live schema; live ends right at baseline end (no gap).
        baseline_idx = pd.date_range("2025-01-01", periods=100, freq="5min", tz="UTC")
        baseline = pd.DataFrame(
            {"sensor_a": 1.0, "sensor_a__weight": 1},
            index=baseline_idx,
        )
        baseline.index.name = "timestamp"
        baseline.to_parquet(processed_dir / "panel_1.parquet")

        live_idx = baseline_idx
        live = pd.DataFrame(
            {"sensor_a": 2.0, "sensor_a__weight": 1},
            index=live_idx,
        )
        live.index.name = "timestamp"
        live.to_parquet(live_dir / "panel_1.parquet")

        new_idx = pd.date_range(live_idx[-1] + pd.Timedelta("5min"), periods=5, freq="5min")
        new_df = pd.DataFrame(
            {"sensor_a": 3.0, "sensor_a__weight": 1},
            index=new_idx,
        )
        new_df.index.name = "timestamp"

        DataManager(config).save_processed_data_live({"1": new_df})

        saved = pd.read_parquet(live_dir / "panel_1.parquet").sort_index()
        # Existing (live=2.0) values must NOT be clobbered by new (only late-arriving NaNs are filled).
        assert saved.loc[live_idx[0], "sensor_a"] == 2.0
        # New rows were appended.
        assert saved.index.max() == new_idx.max()
        assert saved.loc[new_idx[0], "sensor_a"] == 3.0


class TestSaveProcessedDataLiveGapClassification:
    """Natural vs fixable gap classification when live already has a >1h gap."""

    def test_natural_gap_in_both_live_and_baseline_keeps_existing_live(self, dm_workspace):
        """If the gap exists in baseline too, classify as natural and don't replace live."""
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        processed_dir = paths["processed_dir"]
        live_dir = processed_dir / "live"
        processed_dir.mkdir()
        live_dir.mkdir()

        # Baseline has a gap at the same place.
        before_idx = pd.date_range("2025-01-01 00:00", periods=20, freq="5min", tz="UTC")
        after_idx = pd.date_range("2025-01-01 04:00", periods=20, freq="5min", tz="UTC")
        baseline_idx = before_idx.append(after_idx)
        baseline = pd.DataFrame(
            {"sensor_a": 1.0, "sensor_a__weight": 1},
            index=baseline_idx,
        )
        baseline.index.name = "timestamp"
        baseline.to_parquet(processed_dir / "panel_1.parquet")

        # Live has the same gap.
        live = pd.DataFrame(
            {"sensor_a": 2.0, "sensor_a__weight": 1},
            index=baseline_idx,
        )
        live.index.name = "timestamp"
        live.to_parquet(live_dir / "panel_1.parquet")

        new_idx = pd.date_range(after_idx[-1] + pd.Timedelta("5min"), periods=3, freq="5min")
        new_df = pd.DataFrame(
            {"sensor_a": 3.0, "sensor_a__weight": 1},
            index=new_idx,
        )
        new_df.index.name = "timestamp"

        DataManager(config).save_processed_data_live({"1": new_df})

        saved = pd.read_parquet(live_dir / "panel_1.parquet").sort_index()
        # Live values (2.0) preserved — natural-gap branch did NOT replace existing_df with baseline.
        assert saved.loc[before_idx[0], "sensor_a"] == 2.0
        # New rows still appended.
        assert saved.index.max() == new_idx.max()

    def test_fixable_gap_only_in_live_replaces_existing_with_full_processed(self, dm_workspace):
        """Gap in live but NOT in baseline → use full processed to fix the gap."""
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        processed_dir = paths["processed_dir"]
        live_dir = processed_dir / "live"
        processed_dir.mkdir()
        live_dir.mkdir()

        # Baseline is gap-free.
        baseline_idx = pd.date_range("2025-01-01 00:00", periods=100, freq="5min", tz="UTC")
        baseline = pd.DataFrame(
            {"sensor_a": 1.0, "sensor_a__weight": 1},
            index=baseline_idx,
        )
        baseline.index.name = "timestamp"
        baseline.to_parquet(processed_dir / "panel_1.parquet")

        # Live has an artificial gap that baseline does NOT have.
        before_idx = pd.date_range("2025-01-01 00:00", periods=20, freq="5min", tz="UTC")
        after_idx = pd.date_range("2025-01-01 04:00", periods=20, freq="5min", tz="UTC")
        live_idx = before_idx.append(after_idx)
        live = pd.DataFrame(
            {"sensor_a": 2.0, "sensor_a__weight": 1},
            index=live_idx,
        )
        live.index.name = "timestamp"
        live.to_parquet(live_dir / "panel_1.parquet")

        new_idx = pd.date_range(baseline_idx[-1] + pd.Timedelta("5min"), periods=3, freq="5min")
        new_df = pd.DataFrame(
            {"sensor_a": 3.0, "sensor_a__weight": 1},
            index=new_idx,
        )
        new_df.index.name = "timestamp"

        DataManager(config).save_processed_data_live({"1": new_df})

        saved = pd.read_parquet(live_dir / "panel_1.parquet").sort_index()
        # The previously missing gap rows are now filled in (from baseline).
        gap_rows = pd.date_range("2025-01-01 01:45", "2025-01-01 03:55", freq="5min", tz="UTC")
        for ts in gap_rows[:3]:
            assert ts in saved.index, f"Expected fixable gap row {ts} to be filled from baseline"

    def test_fixable_gap_missing_baseline_raises(self, dm_workspace):
        """Live has a gap, but baseline is gone → can't verify → RuntimeError."""
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        processed_dir = paths["processed_dir"]
        live_dir = processed_dir / "live"
        processed_dir.mkdir()
        live_dir.mkdir()

        # Live has a gap. NO baseline parquet file at all.
        before_idx = pd.date_range("2025-01-01 00:00", periods=20, freq="5min", tz="UTC")
        after_idx = pd.date_range("2025-01-01 04:00", periods=20, freq="5min", tz="UTC")
        live_idx = before_idx.append(after_idx)
        live = pd.DataFrame(
            {"sensor_a": 2.0, "sensor_a__weight": 1},
            index=live_idx,
        )
        live.index.name = "timestamp"
        live.to_parquet(live_dir / "panel_1.parquet")

        new_idx = pd.date_range(after_idx[-1] + pd.Timedelta("5min"), periods=3, freq="5min")
        new_df = pd.DataFrame(
            {"sensor_a": 3.0, "sensor_a__weight": 1},
            index=new_idx,
        )
        new_df.index.name = "timestamp"

        with pytest.raises(RuntimeError, match="Cannot verify gaps"):
            DataManager(config).save_processed_data_live({"1": new_df})


# ---------------------------------------------------------------------------
# load_processed_data_live
# ---------------------------------------------------------------------------


class TestLoadProcessedDataLive:
    def test_returns_empty_df_for_missing_panel(self, dm_workspace):
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        # No live dir at all.
        loaded = DataManager(config).load_processed_data_live()
        assert "1" in loaded
        assert loaded["1"].empty

    def test_round_trips_via_live_save(self, dm_workspace):
        config, paths = dm_workspace
        config["panels"]["panel_ids"] = ["1"]
        processed_dir = paths["processed_dir"]
        processed_dir.mkdir()

        baseline = _make_df("2025-01-01", periods=200, value=1.0)
        baseline.to_parquet(processed_dir / "panel_1.parquet")

        new_df = _make_df(str(baseline.index[-1] + pd.Timedelta("5min")), periods=5, value=2.0)
        DataManager(config).save_processed_data_live({"1": new_df})

        loaded = DataManager(config).load_processed_data_live()
        assert not loaded["1"].empty
        assert loaded["1"].index.max() == new_df.index.max()
