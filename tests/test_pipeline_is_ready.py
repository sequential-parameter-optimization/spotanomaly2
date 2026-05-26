# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Pipeline.is_ready() — the live-mode readiness gate.

These tests parametrise over (panel data presence, panel freshness,
model presence, model freshness) and assert the corresponding verdict.
They exist because `is_ready()` is the single safeguard that prevents
``live()`` from scoring against stale models or missing data, and the
existing test suite only mock-traces around it.
"""

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from spotanomaly2.application.pipeline import Pipeline
from spotanomaly2.infrastructure import storage


@pytest.fixture
def ready_workspace(tmp_path, sample_config):
    """Build a tmp workspace with fresh panel data and a fresh model file.

    Returns a (config, processed_dir, models_dir) tuple where every
    `is_ready()` precondition is met out of the box.  Individual tests
    mutate the workspace to flip each precondition and assert the change.
    """
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    sample_config["paths"]["processed_dir"] = str(processed_dir)
    sample_config["paths"]["models_dir"] = str(models_dir)
    sample_config["panels"]["panel_ids"] = ["1"]
    sample_config["detect"]["fc_model_name"] = "LightGBM"

    # Fresh panel parquet: now-ish timestamp index.
    now = pd.Timestamp.now(tz="UTC").floor("min")
    idx = pd.date_range(end=now, periods=100, freq="5min", tz="UTC")
    df = pd.DataFrame({"sensor_a": range(100)}, index=idx)
    storage.save_panel_parquet(df, processed_dir, "1")

    # Fresh model directory + file.
    model_dir = models_dir / "20251231_120000"
    model_dir.mkdir()
    model_file = model_dir / "LightGBM_fc_model_panel_1.pkl"
    model_file.write_bytes(b"\x00")  # contents irrelevant; only mtime matters

    return sample_config, processed_dir, models_dir, model_file


class TestIsReadyHappyPath:
    def test_returns_true_when_everything_fresh(self, ready_workspace):
        config, *_ = ready_workspace
        assert Pipeline(config).is_ready() is True


class TestIsReadyDataPreconditions:
    def test_returns_false_when_panel_parquet_missing(self, ready_workspace):
        config, processed_dir, _, _ = ready_workspace
        (processed_dir / "panel_1.parquet").unlink()
        assert Pipeline(config).is_ready() is False

    def test_returns_false_when_panel_data_stale(self, ready_workspace):
        """Panel whose latest row is older than max_age_days must fail."""
        config, processed_dir, _, _ = ready_workspace
        # Overwrite with data from 10 days ago.
        old_end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
        idx = pd.date_range(end=old_end, periods=100, freq="5min", tz="UTC")
        df = pd.DataFrame({"sensor_a": range(100)}, index=idx)
        storage.save_panel_parquet(df, processed_dir, "1")
        assert Pipeline(config).is_ready(max_age_days=7) is False

    def test_returns_true_for_panel_data_just_under_threshold(self, ready_workspace):
        config, processed_dir, _, _ = ready_workspace
        recent_end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=6)
        idx = pd.date_range(end=recent_end, periods=100, freq="5min", tz="UTC")
        df = pd.DataFrame({"sensor_a": range(100)}, index=idx)
        storage.save_panel_parquet(df, processed_dir, "1")
        assert Pipeline(config).is_ready(max_age_days=7) is True

    def test_returns_false_when_panel_dataframe_empty(self, ready_workspace):
        config, processed_dir, _, _ = ready_workspace
        empty = pd.DataFrame({"sensor_a": []}, index=pd.DatetimeIndex([], tz="UTC"))
        storage.save_panel_parquet(empty, processed_dir, "1")
        assert Pipeline(config).is_ready() is False

    def test_returns_false_when_any_of_multiple_panels_missing(self, ready_workspace):
        config, processed_dir, _, _ = ready_workspace
        config["panels"]["panel_ids"] = ["1", "2"]
        # Panel 1 exists from fixture; panel 2 does not.
        assert Pipeline(config).is_ready() is False


class TestIsReadyModelPreconditions:
    def test_returns_false_when_models_dir_missing(self, ready_workspace, tmp_path):
        config, _, models_dir, _ = ready_workspace
        # Remove the entire models_dir.
        for child in models_dir.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(models_dir.glob("*"), reverse=True):
            child.rmdir()
        models_dir.rmdir()
        assert Pipeline(config).is_ready() is False

    def test_returns_false_when_no_timestamped_dirs(self, ready_workspace):
        config, _, models_dir, _ = ready_workspace
        # Remove the timestamped subdir, leave models_dir empty.
        for child in models_dir.iterdir():
            for f in child.iterdir():
                f.unlink()
            child.rmdir()
        assert Pipeline(config).is_ready() is False

    def test_returns_false_when_no_matching_model_files(self, ready_workspace):
        config, _, _, model_file = ready_workspace
        # Rename the model so it no longer matches the expected pattern.
        model_file.rename(model_file.parent / "Other_fc_model_panel_1.pkl")
        config["detect"]["fc_model_name"] = "LightGBM"
        assert Pipeline(config).is_ready() is False

    def test_returns_false_when_model_file_stale(self, ready_workspace):
        config, _, _, model_file = ready_workspace
        # Backdate the model mtime by 10 days.
        ten_days_ago = time.time() - 10 * 86400
        os.utime(model_file, (ten_days_ago, ten_days_ago))
        assert Pipeline(config).is_ready(max_age_days=7) is False

    def test_returns_true_when_model_just_under_threshold(self, ready_workspace):
        config, _, _, model_file = ready_workspace
        six_days_ago = time.time() - 6 * 86400
        os.utime(model_file, (six_days_ago, six_days_ago))
        assert Pipeline(config).is_ready(max_age_days=7) is True


class TestIsReadyMaxAgeOverride:
    @pytest.mark.parametrize(
        "max_age_days, expected",
        [
            (1, False),
            (3, False),
            (5, True),
            (30, True),
        ],
    )
    def test_threshold_drives_verdict(self, ready_workspace, max_age_days, expected):
        """Panel data and model are both ~4 days old; verdict flips around 4."""
        config, processed_dir, _, model_file = ready_workspace
        four_days_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=4)
        idx = pd.date_range(end=four_days_ago, periods=100, freq="5min", tz="UTC")
        df = pd.DataFrame({"sensor_a": range(100)}, index=idx)
        storage.save_panel_parquet(df, processed_dir, "1")
        ts = time.time() - 4 * 86400
        os.utime(model_file, (ts, ts))

        assert Pipeline(config).is_ready(max_age_days=max_age_days) is expected
