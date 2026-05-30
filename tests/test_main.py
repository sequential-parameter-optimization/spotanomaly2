# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the spotanomaly2 CLI entry point (main.py).

These tests aggressively mock Pipeline and LiveMonitor so they exercise argparse
plumbing, config loading, flag injection, and command dispatch without actually
running pipeline stages.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from spotanomaly2.main import build_parser, main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config_file(tmp_path: Path, sample_config: dict) -> Path:
    """Write the sample_config dict to a YAML file and return its path."""
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_config, f)
    return cfg_path


# ---------------------------------------------------------------------------
# (1) build_parser smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["download"],
        ["process"],
        ["train"],
        ["detect"],
        ["tune"],
        ["live"],
    ],
)
def test_build_parser_accepts_subcommands(argv):
    """Every documented subcommand parses without raising."""
    parser = build_parser()
    args = parser.parse_args(argv)
    assert args.command == argv[0]


def test_build_parser_top_level_help_exits_zero():
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["download", "--help"],
        ["detect", "--help"],
        ["tune", "--help"],
        ["live", "--help"],
    ],
)
def test_subcommand_help_exits_zero(argv):
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 0


# ---------------------------------------------------------------------------
# (2) --config flag wiring
# ---------------------------------------------------------------------------


def test_config_flag_invokes_load_config_with_path(tmp_config_file, sample_config):
    """`main(["detect", "--config", path])` must call load_config(path, ...).

    Pipeline is mocked so nothing actually runs.
    """
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline.return_value = MagicMock()

        rc = main(["detect", "--config", str(tmp_config_file)])

        assert rc == 0
        mock_load.assert_called_once()
        # First positional arg should be the path we passed.
        called_path = mock_load.call_args.args[0]
        assert Path(called_path) == tmp_config_file
        # And the Pipeline was constructed with the loaded dict.
        first_arg = mock_pipeline.call_args.args[0]
        assert isinstance(first_arg, dict)


def test_config_missing_returns_nonzero(tmp_path):
    """Pointing --config at a non-existent file should yield non-zero exit."""
    missing = tmp_path / "does-not-exist.yaml"
    rc = main(["detect", "--config", str(missing)])
    assert rc != 0


# ---------------------------------------------------------------------------
# (3) --model flag override
# ---------------------------------------------------------------------------


def test_model_flag_injects_into_detect_config(tmp_config_file, sample_config):
    """`--model X` must end up at config['detect']['model_timestamp'] = X."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline.return_value = MagicMock()

        rc = main(["detect", "--config", str(tmp_config_file), "--model", "20251229_172126"])

        assert rc == 0
        passed_config = mock_pipeline.call_args.args[0]
        assert passed_config["detect"]["model_timestamp"] == "20251229_172126"


# ---------------------------------------------------------------------------
# (4) --raw-data-version flag override
# ---------------------------------------------------------------------------


def test_raw_data_version_injects_into_paths(tmp_config_file, sample_config):
    """`--raw-data-version X` should set config['paths']['raw_data_version'] = X."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline.return_value = MagicMock()

        rc = main(
            [
                "detect",
                "--config",
                str(tmp_config_file),
                "--raw-data-version",
                "20260105_174531",
            ]
        )

        assert rc == 0
        passed_config = mock_pipeline.call_args.args[0]
        assert passed_config["paths"]["raw_data_version"] == "20260105_174531"


def test_raw_data_version_on_process(tmp_config_file, sample_config):
    """--raw-data-version also works on `process`."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline.return_value = MagicMock()

        rc = main(["process", "--config", str(tmp_config_file), "--raw-data-version", "20260101_000000"])

        assert rc == 0
        passed_config = mock_pipeline.call_args.args[0]
        assert passed_config["paths"]["raw_data_version"] == "20260101_000000"


# ---------------------------------------------------------------------------
# (5) No --model passes through (preserves YAML default / absence)
# ---------------------------------------------------------------------------


def test_no_model_flag_leaves_config_detect_untouched(tmp_config_file, sample_config):
    """Without --model, detect.model_timestamp must equal the loaded YAML value
    (or remain absent if YAML did not set it)."""
    cfg = dict(sample_config)
    # sample_config has 'detect' but no model_timestamp key — that's our baseline.
    assert "model_timestamp" not in cfg["detect"]

    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline,
    ):
        mock_load.return_value = cfg
        mock_pipeline.return_value = MagicMock()

        rc = main(["detect", "--config", str(tmp_config_file)])

        assert rc == 0
        passed_config = mock_pipeline.call_args.args[0]
        assert "model_timestamp" not in passed_config["detect"]


# ---------------------------------------------------------------------------
# Individual stage dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command, method",
    [
        ("download", "download"),
        ("process", "process"),
        ("train", "train"),
        ("detect", "detect"),
    ],
)
def test_stage_dispatch_calls_matching_pipeline_method(tmp_config_file, sample_config, command, method):
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        rc = main([command, "--config", str(tmp_config_file)])

        assert rc == 0
        getattr(mock_pipeline, method).assert_called_once_with()


# ---------------------------------------------------------------------------
# (8) tune subcommand
# ---------------------------------------------------------------------------


def test_tune_subcommand_forwards_panel_and_channel(tmp_config_file, sample_config):
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        rc = main(
            [
                "tune",
                "--config",
                str(tmp_config_file),
                "--panel",
                "1",
                "--channel",
                "channel_1_ph",
            ]
        )

        assert rc == 0
        mock_pipeline.tune.assert_called_once_with(panel_id="1", channel="channel_1_ph")


def test_tune_n_trials_and_n_initial_inject_into_config(tmp_config_file, sample_config):
    """--n-trials and --n-initial mutate config['tune'] before pipeline.tune()."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        rc = main(
            [
                "tune",
                "--config",
                str(tmp_config_file),
                "--n-trials",
                "11",
                "--n-initial",
                "3",
            ]
        )

        assert rc == 0
        passed_config = mock_pipeline_cls.call_args.args[0]
        assert passed_config["tune"]["n_trials"] == 11
        assert passed_config["tune"]["n_initial"] == 3


# ---------------------------------------------------------------------------
# (10) live subcommand: single-shot vs interval mode
# ---------------------------------------------------------------------------


def test_live_single_shot_calls_monitor_run_once(tmp_config_file, sample_config):
    """`live` with no --interval builds a LiveMonitor and calls run_once() once."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline"),
        patch("spotanomaly2.main.LiveMonitor") as mock_monitor_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_monitor = MagicMock()
        mock_monitor_cls.return_value = mock_monitor

        rc = main(["live", "--config", str(tmp_config_file)])

        assert rc == 0
        mock_monitor.run_once.assert_called_once_with()
        mock_monitor.run_monitoring.assert_not_called()


def test_live_with_interval_calls_run_monitoring(tmp_config_file, sample_config):
    """`live --interval 5` routes through LiveMonitor.run_monitoring(5)."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline"),
        patch("spotanomaly2.main.LiveMonitor") as mock_monitor_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_monitor = MagicMock()
        mock_monitor.run_monitoring.return_value = 0
        mock_monitor_cls.return_value = mock_monitor

        rc = main(["live", "--config", str(tmp_config_file), "--interval", "5"])

        assert rc == 0
        mock_monitor.run_monitoring.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# (11) Unknown subcommand
# ---------------------------------------------------------------------------


def test_unknown_subcommand_exits_nonzero():
    """argparse should reject an unknown subcommand with SystemExit != 0."""
    with pytest.raises(SystemExit) as excinfo:
        main(["definitely-not-a-real-command"])
    # argparse uses code 2 for usage errors.
    assert excinfo.value.code != 0


# ---------------------------------------------------------------------------
# (12) Missing required argument
# ---------------------------------------------------------------------------


def test_no_subcommand_exits_nonzero():
    """No subcommand should fail because subparsers is required=True."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0


# ---------------------------------------------------------------------------
# (13) Top-level exception handling
# ---------------------------------------------------------------------------


def test_pipeline_exception_returns_nonzero(tmp_config_file, sample_config):
    """If Pipeline.detect() raises, main should catch it and return 1."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline = MagicMock()
        mock_pipeline.detect.side_effect = RuntimeError("boom")
        mock_pipeline_cls.return_value = mock_pipeline

        rc = main(["detect", "--config", str(tmp_config_file)])

        assert rc != 0


def test_pipeline_construction_exception_returns_nonzero(tmp_config_file, sample_config):
    """Failure constructing Pipeline should also return non-zero."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline_cls.side_effect = ValueError("bad config")

        rc = main(["detect", "--config", str(tmp_config_file)])

        assert rc != 0
