# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end round-trip across the tune -> config -> train seam (no mocks).

``Pipeline.tune`` runs a real (tiny) SpotOptim search, then writes the winning
model/params/lags into a per-panel ``channel_models`` YAML. ``Pipeline.train``
then reads that YAML back and must actually fit the channel with the tuned
model. Nothing exercised that hand-off before: the writer
(``update_channel_configs``) and reader (``load_panel_channel_config``) were
only tested in isolation, so a format drift between them would have slipped
through.

The search is kept to a single channel, single model, and 2 trials to stay
fast.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from spotanomaly2.application.config import load_panel_channel_config
from spotanomaly2.application.data_manager import DataManager
from spotanomaly2.application.pipeline import Pipeline


def _make_processed_panel(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    idx.name = "timestamp"
    t = np.arange(n)
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "sensor_a": 10.0 + 3.0 * np.sin(2 * np.pi * t / 96) + rng.standard_normal(n) * 0.3,
            "sensor_a__weight": 1.0,
            "sensor_b": 20.0 + 2.0 * np.cos(2 * np.pi * t / 48) + rng.standard_normal(n) * 0.3,
            "sensor_b__weight": 1.0,
        },
        index=idx,
    )


def test_tune_writes_config_that_train_consumes(sample_config, tmp_path):
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
    cfg["train"]["split"] = {"train": 70, "test": 15, "score": 15}
    cfg["train"]["lags"] = 6
    channel_yaml = tmp_path / "panel_1.yaml"
    cfg["train"]["channel_config_files"] = {"1": str(channel_yaml)}
    # A non-empty search space is required so SpotOptim has >=2 distinct design
    # points to fit its surrogate. A single candidate model keeps the winner
    # deterministic, so we can assert exactly which model train must fit.
    cfg["tune"] = {
        "n_trials": 3,
        "n_initial": 2,
        "metric": "mean_absolute_error",
        "models": {"LightGBM": {"num_leaves": [8, 31]}},
        "output_dir": str(tmp_path / "tuning_results"),
    }

    DataManager(cfg).save_processed_data({"1": _make_processed_panel()})
    pipeline = Pipeline(cfg)

    # --- tune: real search, then writeback into the channel YAML ---
    pipeline.tune(channel="sensor_a")

    assert channel_yaml.exists(), "tune did not write the channel config"
    written = load_panel_channel_config("1", cfg)
    tuned = written["channels"]["sensor_a"]
    assert tuned["model"] == "LightGBM"
    tuned_lags = tuned["best_lags"]
    assert tuned_lags is not None

    # --- train: must read the tuned spec back and fit with it ---
    _eval_df, ts = pipeline.train()["1"]
    model = joblib.load(Path(cfg["paths"]["models_dir"]) / ts / "fc_model_panel_1.pkl")

    spec = model["channel_models"]["sensor_a"]
    assert spec["model"] == "LightGBM"

    # The lags train fit with must be exactly what tune wrote (normalised to a
    # comparable shape — YAML may store a list or a scalar).
    def _norm(lags):
        if isinstance(lags, (list, tuple)):
            return [int(x) for x in lags]
        return int(lags)

    assert _norm(spec["lags"]) == _norm(tuned_lags)
