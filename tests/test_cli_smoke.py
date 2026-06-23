# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI-level smoke test: real `main(["train"])` / `main(["detect"])`, no mocks.

``test_main.py`` patches out ``Pipeline`` to characterise argument parsing and
dispatch, so it cannot catch wiring bugs in the real command path — e.g. a
stage method whose return contract is broken. This module runs the actual CLI
entry point against synthetic processed data on disk and asserts the commands
exit 0 and leave the expected artifacts. It is the entry point users actually
invoke, so it is the right altitude for a "the basics still run" guard.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spotanomaly2.application.data_manager import DataManager
from spotanomaly2.main import main


def _make_processed_panel(n: int = 800) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    idx.name = "timestamp"
    t = np.arange(n)
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "sensor_a": 10.0 + 3.0 * np.sin(2 * np.pi * t / 96) + rng.standard_normal(n) * 0.3,
            "sensor_a__weight": 1.0,
            "sensor_b": 20.0 + 2.0 * np.cos(2 * np.pi * t / 48) + rng.standard_normal(n) * 0.3,
            "sensor_b__weight": 1.0,
        },
        index=idx,
    )


def _write_config_and_data(sample_config, tmp_path) -> Path:
    """Materialise a runnable config YAML + processed parquet under tmp_path."""
    cfg = {
        **sample_config,
        "paths": {
            **sample_config["paths"],
            "processed_dir": str(tmp_path / "processed"),
            "models_dir": str(tmp_path / "models"),
            "results_dir": str(tmp_path / "results"),
        },
    }
    cfg["panels"]["panel_ids"] = ["1"]
    cfg["train"]["split"] = {"train": 60, "val": 10, "test": 30}
    cfg["train"]["lags"] = 6
    cfg["detect"]["hist_window"] = 40

    DataManager(cfg).save_processed_data({"1": _make_processed_panel()})

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path


def test_cli_train_then_detect_exit_zero(sample_config, tmp_path):
    config_path = _write_config_and_data(sample_config, tmp_path)

    # train: a broken stage-return contract (the kind of bug mocks hide) would
    # surface here as a non-zero exit code.
    rc_train = main(["train", "--config", str(config_path)])
    assert rc_train == 0, "`spotanomaly2 train` exited non-zero"

    models_root = tmp_path / "models"
    model_files = list(models_root.glob("*/fc_model_panel_1.pkl"))
    assert model_files, "train command produced no model artifact"

    # detect: loads the model train just wrote and runs scoring end to end.
    rc_detect = main(["detect", "--config", str(config_path)])
    assert rc_detect == 0, "`spotanomaly2 detect` exited non-zero"


def test_cli_unknown_config_path_exits_nonzero(tmp_path):
    rc = main(["detect", "--config", str(tmp_path / "does_not_exist.yaml")])
    assert rc == 1
