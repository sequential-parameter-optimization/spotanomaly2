# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the spotanomaly2 CLI entry point (main.py).

These tests aggressively mock Pipeline (and signal/asyncio for live mode) so
they exercise argparse plumbing, config loading, flag injection, and command
dispatch without actually running pipeline stages.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from spotanomaly2 import main as main_module
from spotanomaly2.main import build_parser, main, run_live_monitoring


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
        ["all"],
        ["tune"],
        ["live"],
        ["benchmark"],
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
        ["all", "--help"],
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


def test_model_flag_on_all_subcommand_injects(tmp_config_file, sample_config):
    """`--model` should also work on the 'all' subcommand."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline.return_value = MagicMock()

        rc = main(
            [
                "all",
                "--config",
                str(tmp_config_file),
                "--predict-only",
                "--model",
                "20251229_172126",
            ]
        )

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

        rc = main(
            ["process", "--config", str(tmp_config_file), "--raw-data-version", "20260101_000000"]
        )

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
# (6) & (7) `all` flags: --predict-only, --skip-download
# ---------------------------------------------------------------------------


def test_all_predict_only_flag_propagates(tmp_config_file, sample_config):
    """`all --predict-only` must call pipeline.run_all(predict_only=True)."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        rc = main(["all", "--config", str(tmp_config_file), "--predict-only"])

        assert rc == 0
        mock_pipeline.run_all.assert_called_once_with(skip_download=False, predict_only=True)


def test_all_skip_download_flag_propagates(tmp_config_file, sample_config):
    """`all --skip-download` must call pipeline.run_all(skip_download=True)."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        rc = main(["all", "--config", str(tmp_config_file), "--skip-download"])

        assert rc == 0
        mock_pipeline.run_all.assert_called_once_with(skip_download=True, predict_only=False)


def test_all_no_flags_defaults(tmp_config_file, sample_config):
    """`all` with no flags should call run_all(skip_download=False, predict_only=False)."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        rc = main(["all", "--config", str(tmp_config_file)])

        assert rc == 0
        mock_pipeline.run_all.assert_called_once_with(skip_download=False, predict_only=False)


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
def test_stage_dispatch_calls_matching_pipeline_method(
    tmp_config_file, sample_config, command, method
):
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
# (9) benchmark subcommand dispatch
# ---------------------------------------------------------------------------


def test_benchmark_dispatches_to_research_run(tmp_config_file, sample_config):
    """`benchmark` should call spotanomaly2.research.cli.run and return its exit code."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
        patch("spotanomaly2.research.cli.run") as mock_run_benchmark,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline_cls.return_value = MagicMock()
        mock_run_benchmark.return_value = 0

        rc = main(["benchmark", "--config", str(tmp_config_file)])

        assert rc == 0
        mock_run_benchmark.assert_called_once()


def test_benchmark_propagates_nonzero_exit_code(tmp_config_file, sample_config):
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
        patch("spotanomaly2.research.cli.run") as mock_run_benchmark,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline_cls.return_value = MagicMock()
        mock_run_benchmark.return_value = 7

        rc = main(["benchmark", "--config", str(tmp_config_file)])

        assert rc == 7


# ---------------------------------------------------------------------------
# (10) live subcommand: single-shot vs interval mode
# ---------------------------------------------------------------------------


def test_live_single_shot_calls_pipeline_live(tmp_config_file, sample_config):
    """`live` with no --interval calls pipeline.live() exactly once."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        rc = main(["live", "--config", str(tmp_config_file)])

        assert rc == 0
        mock_pipeline.live.assert_called_once_with()


def test_live_with_interval_calls_run_live_monitoring(tmp_config_file, sample_config):
    """`live --interval 5` routes through run_live_monitoring."""
    with (
        patch("spotanomaly2.main.load_config") as mock_load,
        patch("spotanomaly2.main.Pipeline") as mock_pipeline_cls,
        patch("spotanomaly2.main.run_live_monitoring") as mock_run_live,
    ):
        mock_load.return_value = dict(sample_config)
        mock_pipeline_cls.return_value = MagicMock()
        mock_run_live.return_value = 0

        rc = main(["live", "--config", str(tmp_config_file), "--interval", "5"])

        assert rc == 0
        mock_run_live.assert_called_once()
        # Second positional arg is the interval in minutes.
        assert mock_run_live.call_args.args[1] == 5


def test_run_live_monitoring_rejects_interval_lt_1():
    """Sanity guard: interval must be at least 1."""
    logger = MagicMock()
    pipeline = MagicMock()
    rc = run_live_monitoring(pipeline, 0, logger)
    assert rc != 0
    logger.error.assert_called()


def test_run_live_monitoring_loops_until_signal(monkeypatch):
    """run_live_monitoring should call pipeline.live() at least once and stop cleanly.

    We patch signal.signal, time.sleep, and LiveReportServer to make the loop
    deterministic and fast. We force the loop to terminate after one iteration
    by mutating the `running` flag via a side-effect on pipeline.live().
    """
    pipeline = MagicMock()
    pipeline.config = {"paths": {"results_dir": "/tmp/results-test"}, "report": {"enabled": False}}
    logger = MagicMock()

    # Capture and stub signal handlers (don't really register them).
    monkeypatch.setattr(main_module.signal, "signal", MagicMock())
    # Skip real sleeping.
    monkeypatch.setattr(main_module.time, "sleep", MagicMock())

    # The loop reads `time.time()` to decide when to wake up. Returning a huge
    # value makes the inner wait loop fall through immediately on first check.
    real_time = main_module.time.time

    call_count = {"n": 0}

    def fake_time():
        call_count["n"] += 1
        # First few calls real, then jump far in the future to break the wait.
        return real_time() + 10_000 * call_count["n"]

    monkeypatch.setattr(main_module.time, "time", fake_time)

    # Side effect: after first pipeline.live(), raise KeyboardInterrupt so the
    # outer while-loop exits via its `except KeyboardInterrupt: break`.
    pipeline.live.side_effect = KeyboardInterrupt()

    rc = run_live_monitoring(pipeline, 1, logger)

    assert rc == 0
    assert pipeline.live.call_count == 1


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
