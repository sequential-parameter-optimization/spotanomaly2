# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end smoke test for the core spotanomaly2 chain: train -> detect.

Unlike ``test_pipeline.py`` / ``test_pipeline_integration.py`` (which mock out
the domain layer to characterise orchestration glue), this module runs the real
``ModelTrainer`` and ``AnomalyDetector`` against synthetic processed data on
disk. It is the "do the basics still work together" guard:

  1. processed panels are written to disk,
  2. ``Pipeline.train()`` trains a real forecaster and persists a model pickle,
  3. ``Pipeline.detect()`` loads that exact pickle and produces well-formed,
     finite anomaly scores and binary flags.

No domain class is mocked. If the train->detect seam breaks (e.g. a model
artifact the detector can't consume), this test fails where the unit tests —
each of which exercises only one half of the seam — would stay green.
"""

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from spotanomaly2.application.data_manager import DataManager
from spotanomaly2.application.pipeline import Pipeline
from spotanomaly2.domain.exceptions import InsufficientDataException
from spotanomaly2.domain.processing.data_processor import DataProcessor


def _make_processed_panel(n: int = 800, spike: bool = False) -> pd.DataFrame:
    """Two-channel periodic signal at 5-min cadence with all-observed weights.

    A periodic-plus-noise signal (rather than pure noise) gives the forecaster
    something learnable, so the residuals fed to the scorer are well-behaved.
    ``__weight`` columns mark every row as observed so the detector's
    imputed-row exclusion keeps the full window.
    """
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    idx.name = "timestamp"
    t = np.arange(n)
    rng = np.random.default_rng(7)
    sensor_a = 10.0 + 3.0 * np.sin(2 * np.pi * t / 96) + rng.standard_normal(n) * 0.3
    sensor_b = 20.0 + 2.0 * np.cos(2 * np.pi * t / 48) + rng.standard_normal(n) * 0.3

    if spike:
        # Obvious, late, out-of-sample anomaly in sensor_a (≈ +15, ~50σ).
        sensor_a[n - 10 : n - 7] += 15.0

    return pd.DataFrame(
        {
            "sensor_a": sensor_a,
            "sensor_a__weight": 1.0,
            "sensor_b": sensor_b,
            "sensor_b__weight": 1.0,
        },
        index=idx,
    )


@pytest.fixture
def e2e_config(sample_config, tmp_path):
    """sample_config pointed at tmp dirs, sized so detection has room to run."""
    config = copy.deepcopy(sample_config)
    config["paths"]["processed_dir"] = str(tmp_path / "processed")
    config["paths"]["models_dir"] = str(tmp_path / "models")
    config["paths"]["results_dir"] = str(tmp_path / "results")
    config["panels"]["panel_ids"] = ["1"]
    # Reserve ~30% of the series as the scorer's unseen window (the configured
    # ``test`` slice) so the scorer fit/eval split (hist_window) fits
    # comfortably inside it. train + val = 70% is seen by trainer/tuner.
    config["train"]["split"] = {"train": 60, "val": 10, "test": 30}
    config["train"]["lags"] = 6
    config["detect"]["hist_window"] = 40
    return config


def test_train_then_detect_end_to_end(e2e_config, tmp_path):
    """Real train -> persisted model -> real detect, with well-formed output."""
    processed = {"1": _make_processed_panel()}
    DataManager(e2e_config).save_processed_data(processed)

    pipeline = Pipeline(e2e_config)

    # --- train ---
    train_results = pipeline.train()
    assert set(train_results) == {"1"}
    eval_df, timestamp = train_results["1"]
    assert isinstance(eval_df, pd.DataFrame)
    assert {"val_rmse", "val_mae", "test_rmse", "test_mae"}.issubset(eval_df.columns)

    # The model pickle the detector will load must actually be on disk.
    model_path = Path(e2e_config["paths"]["models_dir"]) / timestamp / "fc_model_panel_1.pkl"
    assert model_path.exists(), f"expected trained model at {model_path}"

    # --- detect (loads the model just trained) ---
    results = pipeline.detect()
    assert set(results) == {"1"}

    scores_df, flags_df, pred_df, _contributions = results["1"]

    # Scores are present and finite (no NaN/Inf leaking through the seam).
    assert "anomaly_score_normalized" in scores_df.columns
    assert len(scores_df) > 0
    assert np.isfinite(scores_df["anomaly_score_normalized"].to_numpy()).all()

    # Flags are strictly binary and index-aligned with the scores.
    flag_vals = set(np.unique(flags_df["anomaly_flag"].to_numpy()))
    assert flag_vals.issubset({0, 1})
    assert len(flags_df) == len(scores_df)

    # Predictions cover the trained channels.
    assert {"sensor_a", "sensor_b"}.issubset(set(pred_df.columns))


def test_injected_spike_is_flagged_end_to_end(e2e_config):
    """A blatant out-of-sample spike should produce at least one anomaly flag."""
    processed = {"1": _make_processed_panel(spike=True)}
    DataManager(e2e_config).save_processed_data(processed)

    pipeline = Pipeline(e2e_config)
    pipeline.train()
    results = pipeline.detect()

    _scores, flags_df, _pred, _contrib = results["1"]
    assert int(flags_df["anomaly_flag"].sum()) >= 1, "blatant injected spike was not flagged"


# ---------------------------------------------------------------------------
# Tier 1 #2 — process -> train -> detect through the REAL DataProcessor.
# Guards the processed-schema contract: that what `process` emits is exactly
# what `train`/`detect` can consume (weight columns, column naming, resample).
# ---------------------------------------------------------------------------


def _make_raw_panel(n: int = 2400) -> pd.DataFrame:
    """Raw-shaped panel (pre-process): flow + temperature + maintenance flag."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    idx.name = "timestamp"
    t = np.arange(n)
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "channel_0_flow_primary": 10.0 + 3.0 * np.sin(2 * np.pi * t / 480) + rng.standard_normal(n) * 0.3,
            "channel_0_temperature_1": 15.0 + rng.standard_normal(n) * 0.3,
            "channel_0_maintenance_flag": np.zeros(n, dtype=int),
        },
        index=idx,
    )


def test_process_train_detect_full_chain(e2e_config):
    """Run the real DataProcessor, then train + detect on its actual output."""
    processed = DataProcessor(e2e_config).run({"1": _make_raw_panel()})
    assert processed and "1" in processed

    # The processed frame must carry imputation weight columns and at least one
    # numeric target — the contract train relies on.
    proc_df = processed["1"]
    weight_cols = [c for c in proc_df.columns if c.endswith("__weight")]
    target_cols = [c for c in proc_df.columns if not c.endswith("__weight")]
    assert weight_cols, "processed data is missing imputation weight columns"
    assert target_cols, "processed data has no target columns"

    DataManager(e2e_config).save_processed_data(processed)
    pipeline = Pipeline(e2e_config)

    train_results = pipeline.train()
    eval_df, _ts = train_results["1"]
    assert len(eval_df) >= 1  # at least one channel trained

    results = pipeline.detect()
    scores_df = results["1"][0]
    assert np.isfinite(scores_df["anomaly_score_normalized"].to_numpy()).all()


# ---------------------------------------------------------------------------
# Tier 2 #4 — multi-panel: shared training timestamp, per-panel artifacts.
# ---------------------------------------------------------------------------


def test_multi_panel_train_and_detect(e2e_config):
    e2e_config["panels"]["panel_ids"] = ["1", "2"]
    processed = {"1": _make_processed_panel(), "2": _make_processed_panel()}
    DataManager(e2e_config).save_processed_data(processed)

    pipeline = Pipeline(e2e_config)
    train_results = pipeline.train()
    assert set(train_results) == {"1", "2"}

    # train_all_panels stamps every panel with ONE shared timestamp.
    ts_1, ts_2 = train_results["1"][1], train_results["2"][1]
    assert ts_1 == ts_2
    models_dir = Path(e2e_config["paths"]["models_dir"]) / ts_1
    assert (models_dir / "fc_model_panel_1.pkl").exists()
    assert (models_dir / "fc_model_panel_2.pkl").exists()

    results = pipeline.detect()
    assert set(results) == {"1", "2"}
    for pid in ("1", "2"):
        assert np.isfinite(results[pid][0]["anomaly_score_normalized"].to_numpy()).all()


# ---------------------------------------------------------------------------
# Tier 2 #5 — exogenous features flow through train + detect.
# Guards the exog-imputation unification (prepare_channel) end to end.
# ---------------------------------------------------------------------------


def _make_processed_panel_with_exog(n: int = 800) -> pd.DataFrame:
    df = _make_processed_panel(n=n)
    t = np.arange(n)
    # `exogenous_*` columns are auto-detected as features, never scored.
    df["exogenous_temp"] = 12.0 + 2.0 * np.sin(2 * np.pi * t / 120)
    df["exogenous_temp__weight"] = 1.0
    return df


def test_exogenous_features_end_to_end(e2e_config):
    DataManager(e2e_config).save_processed_data({"1": _make_processed_panel_with_exog()})

    pipeline = Pipeline(e2e_config)
    train_results = pipeline.train()
    eval_df, ts = train_results["1"]

    # Exogenous column is a feature, not a scored target.
    assert "exogenous_temp" not in eval_df.index
    assert {"sensor_a", "sensor_b"}.issubset(set(eval_df.index))

    model = joblib.load(Path(e2e_config["paths"]["models_dir"]) / ts / "fc_model_panel_1.pkl")
    assert "exogenous_temp" in model["exog_columns"]

    results = pipeline.detect()
    assert np.isfinite(results["1"][0]["anomaly_score_normalized"].to_numpy()).all()


# ---------------------------------------------------------------------------
# Tier 2 #6 — differentiation: detect predictions land in raw-y space for both
# differentiation = 0 and = 1 (guards _predict_one_step_integrated).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("diff_order", [0, 1])
def test_detect_predictions_are_in_raw_space(e2e_config, diff_order):
    e2e_config["train"]["differentiation"] = diff_order
    DataManager(e2e_config).save_processed_data({"1": _make_processed_panel()})

    pipeline = Pipeline(e2e_config)
    pipeline.train()
    results = pipeline.detect()
    pred_df = results["1"][2]

    # sensor_a oscillates around 10; a Δy model that wasn't integrated back
    # would predict around 0. Assert predictions sit near the raw level.
    assert abs(float(pred_df["sensor_a"].mean()) - 10.0) < 5.0
    assert abs(float(pred_df["sensor_b"].mean()) - 20.0) < 5.0


# ---------------------------------------------------------------------------
# Tier 3 #8 — determinism: a fixed seed reproduces identical eval metrics.
# Uses Ridge (fully deterministic) to avoid tree-library nondeterminism.
# ---------------------------------------------------------------------------


def test_training_is_deterministic_under_fixed_seed(e2e_config):
    e2e_config["train"]["fallback_model"] = "Ridge"
    panel = _make_processed_panel()

    def _train_eval():
        cfg = copy.deepcopy(e2e_config)
        DataManager(cfg).save_processed_data({"1": panel.copy()})
        return Pipeline(cfg).train()["1"][0]

    eval_a = _train_eval()
    eval_b = _train_eval()
    np.testing.assert_allclose(eval_a["val_rmse"].to_numpy(), eval_b["val_rmse"].to_numpy())
    np.testing.assert_allclose(eval_a["val_mae"].to_numpy(), eval_b["val_mae"].to_numpy())
    np.testing.assert_allclose(eval_a["test_rmse"].to_numpy(), eval_b["test_rmse"].to_numpy())
    np.testing.assert_allclose(eval_a["test_mae"].to_numpy(), eval_b["test_mae"].to_numpy())


# ---------------------------------------------------------------------------
# Tier 3 #9 — degenerate inputs fail gracefully, not with stack traces.
# ---------------------------------------------------------------------------


def test_fully_imputed_channel_is_skipped(e2e_config):
    """A channel that is entirely imputed must be dropped, not crash training."""
    e2e_config["train"]["exclude_imputed_training_samples"] = True
    panel = _make_processed_panel()
    panel["sensor_b__weight"] = 0.0  # every sensor_b row is imputed

    DataManager(e2e_config).save_processed_data({"1": panel})
    _eval_df, ts = Pipeline(e2e_config).train()["1"]

    model = joblib.load(Path(e2e_config["paths"]["models_dir"]) / ts / "fc_model_panel_1.pkl")
    assert "sensor_a" in model["forecasters"]
    assert "sensor_b" not in model["forecasters"]


def test_detect_without_enough_unseen_data_raises_typed_error(e2e_config):
    """Too little post-training data must raise the typed InsufficientDataException."""
    # A 5% test window on 300 rows leaves ~15 unseen rows, far short of
    # hist_window=40 — detection cannot form a valid window.
    e2e_config["train"]["split"] = {"train": 85, "val": 10, "test": 5}
    DataManager(e2e_config).save_processed_data({"1": _make_processed_panel(n=300)})

    pipeline = Pipeline(e2e_config)
    pipeline.train()
    with pytest.raises(InsufficientDataException):
        pipeline.detect()
