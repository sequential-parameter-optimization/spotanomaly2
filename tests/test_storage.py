# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for infrastructure/storage.py.

These tests cover the parquet I/O helpers and the timestamped-directory
resolution that ``--model`` / ``--raw-data-version`` rely on.  Behaviour
exercised here is small, pure (filesystem-only), and deterministic.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from spotanomaly2.infrastructure import storage


# ---------------------------------------------------------------------------
# generate_timestamp


class TestGenerateTimestamp:
    def test_matches_expected_format(self):
        ts = storage.generate_timestamp()
        assert re.fullmatch(r"\d{8}_\d{6}", ts), f"unexpected timestamp shape: {ts!r}"

    def test_round_trips_through_strptime(self):
        ts = storage.generate_timestamp()
        # Must be parseable as YYYYMMDD_HHMMSS, so downstream sort-by-name works.
        parsed = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        assert parsed.year >= 2025


# ---------------------------------------------------------------------------
# find_latest_timestamped_dir


class TestFindLatestTimestampedDir:
    def test_returns_chronologically_latest(self, tmp_path):
        # Three valid timestamps; assert the newest wins regardless of mtime.
        for name in ("20240101_000000", "20251231_235959", "20250630_120000"):
            (tmp_path / name).mkdir()
        latest = storage.find_latest_timestamped_dir(tmp_path)
        assert latest.name == "20251231_235959"

    def test_ignores_non_timestamped_dirs(self, tmp_path):
        # The current regex only checks digit shape, not date validity,
        # so we use clearly non-matching names here.
        (tmp_path / "20251231_235959").mkdir()
        (tmp_path / "scratch").mkdir()
        (tmp_path / "2025-12-31_23-59-59").mkdir()  # wrong separators
        (tmp_path / "20251231").mkdir()  # missing time component
        latest = storage.find_latest_timestamped_dir(tmp_path)
        assert latest.name == "20251231_235959"

    def test_returns_specific_when_timestamp_given(self, tmp_path):
        for name in ("20240101_000000", "20251231_235959"):
            (tmp_path / name).mkdir()
        chosen = storage.find_latest_timestamped_dir(tmp_path, model_timestamp="20240101_000000")
        assert chosen.name == "20240101_000000"

    def test_raises_when_specific_timestamp_missing(self, tmp_path):
        (tmp_path / "20251231_235959").mkdir()
        with pytest.raises(FileNotFoundError, match="20240101_000000"):
            storage.find_latest_timestamped_dir(tmp_path, model_timestamp="20240101_000000")

    def test_raises_when_base_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Base directory not found"):
            storage.find_latest_timestamped_dir(tmp_path / "no_such_dir")

    def test_raises_when_no_timestamped_dirs(self, tmp_path):
        (tmp_path / "scratch").mkdir()  # non-matching dir
        with pytest.raises(FileNotFoundError, match="No timestamped directories"):
            storage.find_latest_timestamped_dir(tmp_path)

    def test_ignores_regular_files(self, tmp_path):
        (tmp_path / "20251231_235959").mkdir()
        (tmp_path / "20250101_000000.txt").write_text("not a dir")
        latest = storage.find_latest_timestamped_dir(tmp_path)
        assert latest.name == "20251231_235959"


# ---------------------------------------------------------------------------
# find_latest_model


class TestFindLatestModel:
    def test_returns_model_in_most_recent_dir(self, tmp_path):
        old = tmp_path / "20240101_000000"
        new = tmp_path / "20251231_235959"
        old.mkdir()
        new.mkdir()
        (old / "model_panel_1.pkl").write_bytes(b"\x00")
        (new / "model_panel_1.pkl").write_bytes(b"\x01")
        path = storage.find_latest_model(tmp_path, "model_panel_1.pkl")
        assert path.parent == new

    def test_falls_back_to_older_dir_when_model_only_exists_there(self, tmp_path):
        """If the newest dir is missing the file, scan older dirs."""
        old = tmp_path / "20240101_000000"
        new = tmp_path / "20251231_235959"
        old.mkdir()
        new.mkdir()
        (old / "model_panel_1.pkl").write_bytes(b"\x00")
        # new/ deliberately empty for this panel
        path = storage.find_latest_model(tmp_path, "model_panel_1.pkl")
        assert path.parent == old

    def test_specific_timestamp_returns_that_dir(self, tmp_path):
        old = tmp_path / "20240101_000000"
        new = tmp_path / "20251231_235959"
        old.mkdir()
        new.mkdir()
        (old / "model_panel_1.pkl").write_bytes(b"\x00")
        (new / "model_panel_1.pkl").write_bytes(b"\x01")
        path = storage.find_latest_model(tmp_path, "model_panel_1.pkl", model_timestamp="20240101_000000")
        assert path.parent == old

    def test_specific_timestamp_missing_file_raises(self, tmp_path):
        d = tmp_path / "20251231_235959"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            storage.find_latest_model(tmp_path, "model_panel_1.pkl", model_timestamp="20251231_235959")

    def test_no_timestamped_dirs_raises(self, tmp_path):
        (tmp_path / "scratch").mkdir()
        with pytest.raises(FileNotFoundError, match="No timestamped directories"):
            storage.find_latest_model(tmp_path, "model_panel_1.pkl")

    def test_no_dir_contains_model_raises(self, tmp_path):
        (tmp_path / "20251231_235959").mkdir()
        (tmp_path / "20240101_000000").mkdir()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            storage.find_latest_model(tmp_path, "absent.pkl")


# ---------------------------------------------------------------------------
# parquet round-trip


class TestParquetRoundTrip:
    def test_save_then_load_returns_equal_frame(self, tmp_path):
        idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
        df = pd.DataFrame({"a": range(10), "b": [x * 0.5 for x in range(10)]}, index=idx)
        path = tmp_path / "x.parquet"
        storage.save_parquet(df, path)
        loaded = storage.load_parquet(path)
        # Parquet does not preserve DatetimeIndex.freq, so compare without it.
        pd.testing.assert_frame_equal(df, loaded, check_freq=False)

    def test_save_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        path = nested / "out.parquet"
        df = pd.DataFrame({"v": [1, 2, 3]})
        storage.save_parquet(df, path)
        assert path.exists()
        assert nested.is_dir()

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Parquet file not found"):
            storage.load_parquet(tmp_path / "nope.parquet")

    def test_panel_parquet_uses_panel_filename(self, tmp_path):
        df = pd.DataFrame({"v": [1, 2]})
        storage.save_panel_parquet(df, tmp_path, "42")
        assert (tmp_path / "panel_42.parquet").exists()
        pd.testing.assert_frame_equal(storage.load_panel_parquet(tmp_path, "42"), df)

    def test_versioned_panel_load_picks_latest(self, tmp_path):
        old = tmp_path / "20240101_000000"
        new = tmp_path / "20251231_235959"
        old.mkdir()
        new.mkdir()
        df_old = pd.DataFrame({"v": [0]})
        df_new = pd.DataFrame({"v": [9]})
        storage.save_panel_parquet(df_old, old, "1")
        storage.save_panel_parquet(df_new, new, "1")
        loaded = storage.load_panel_parquet_versioned(tmp_path, "1")
        pd.testing.assert_frame_equal(loaded, df_new)

    def test_versioned_panel_load_with_explicit_version(self, tmp_path):
        old = tmp_path / "20240101_000000"
        new = tmp_path / "20251231_235959"
        old.mkdir()
        new.mkdir()
        df_old = pd.DataFrame({"v": [0]})
        df_new = pd.DataFrame({"v": [9]})
        storage.save_panel_parquet(df_old, old, "1")
        storage.save_panel_parquet(df_new, new, "1")
        loaded = storage.load_panel_parquet_versioned(tmp_path, "1", version="20240101_000000")
        pd.testing.assert_frame_equal(loaded, df_old)


# ---------------------------------------------------------------------------
# raw metadata JSON


class TestRawMetadata:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "meta.json"
        metadata = {"start_date": "2025-01-01", "end_date": "2025-01-31", "n_panels": 3}
        storage.save_raw_metadata(metadata, path)
        assert storage.load_raw_metadata(path) == metadata

    def test_save_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "meta.json"
        storage.save_raw_metadata({"k": "v"}, path)
        assert path.exists()

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            storage.load_raw_metadata(tmp_path / "absent.json")

    def test_saved_file_is_valid_json(self, tmp_path):
        path = tmp_path / "meta.json"
        storage.save_raw_metadata({"a": 1}, path)
        with open(path) as f:
            assert json.load(f) == {"a": 1}


# ---------------------------------------------------------------------------
# ensure_dir


class TestEnsureDir:
    def test_creates_missing_dir(self, tmp_path):
        target = tmp_path / "a" / "b"
        result = storage.ensure_dir(target)
        assert target.is_dir()
        assert result == target

    def test_noop_when_dir_exists(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        # Should not raise.
        result = storage.ensure_dir(target)
        assert result == target
