# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotanomaly2.research.scorer_tuner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.research._spotoptim_helpers import array_to_params, convert_search_space
from spotanomaly2.research.scorer_tuner import tune_scorer


def test_convert_search_space_int_float_factor() -> None:
    space = {
        "n_clusters": (2, 10),
        "learning_rate": (0.001, 0.1, "log10"),
        "covariance": ["full", "diag"],
    }
    bounds, var_type, var_name, var_trans = convert_search_space(space)
    assert var_name == ["n_clusters", "learning_rate", "covariance"]
    assert var_type == ["int", "float", "factor"]
    assert bounds == [(2, 10), (0.001, 0.1), ["full", "diag"]]
    assert var_trans == [None, "log10", None]


def test_convert_search_space_rejects_bad_entry() -> None:
    with pytest.raises(ValueError):
        convert_search_space({"x": "not-a-tuple"})


def test_array_to_params_typed() -> None:
    bounds, var_type, var_name, _ = convert_search_space(
        {"k": (1, 10), "lr": (0.01, 0.5), "kind": ["a", "b", "c"]}
    )
    out = array_to_params(np.array([4.6, 0.123, 1.0]), var_name, var_type, bounds)
    assert out == {"k": 5, "lr": 0.123, "kind": "b"}


@pytest.fixture
def labelled_residual_split() -> dict:
    """Forecaster-style train/test split with strong injected anomalies in test."""
    rng = np.random.default_rng(7)
    n_train, n_test = 600, 200
    idx_tr = pd.RangeIndex(n_train)
    idx_te = pd.RangeIndex(n_train, n_train + n_test)
    y_true_train = pd.DataFrame(rng.normal(size=(n_train, 2)), index=idx_tr, columns=["a", "b"])
    y_pred_train = y_true_train + rng.normal(scale=0.2, size=y_true_train.shape)
    y_true_test = pd.DataFrame(rng.normal(size=(n_test, 2)), index=idx_te, columns=["a", "b"])
    y_pred_test = y_true_test + rng.normal(scale=0.2, size=y_true_test.shape)

    spike_positions = [20, 60, 100, 140, 180]
    for pos in spike_positions:
        y_true_test.iloc[pos, 0] += 15.0
    labels = np.zeros(n_test, dtype=int)
    for pos in spike_positions:
        labels[pos] = 1

    return {
        "y_true_train": y_true_train,
        "y_pred_train": y_pred_train,
        "y_true_test": y_true_test,
        "y_pred_test": y_pred_test,
        "labels": labels,
    }


def test_tuner_finds_better_than_random_kmeans(labelled_residual_split: dict) -> None:
    search_space = {"n_clusters": (2, 8)}
    best_params, trials_df = tune_scorer(
        scorer_name="KMeansScorer",
        search_space=search_space,
        y_true_train=labelled_residual_split["y_true_train"],
        y_pred_train=labelled_residual_split["y_pred_train"],
        y_true_test=labelled_residual_split["y_true_test"],
        y_pred_test=labelled_residual_split["y_pred_test"],
        y_true_labels=labelled_residual_split["labels"],
        n_trials=8,
        n_initial=4,
        random_state=0,
    )
    assert "n_clusters" in best_params
    assert 2 <= best_params["n_clusters"] <= 8
    assert len(trials_df) == 8
    # Best PR-AUC should beat the prevalence baseline (5/200 = 0.025).
    assert trials_df["pr_auc"].max() > 0.05


def test_tuner_handles_norm_scorer_single_param(labelled_residual_split: dict) -> None:
    search_space = {"ord": (1, 4)}
    best_params, trials_df = tune_scorer(
        scorer_name="NormScorer",
        search_space=search_space,
        y_true_train=labelled_residual_split["y_true_train"],
        y_pred_train=labelled_residual_split["y_pred_train"],
        y_true_test=labelled_residual_split["y_true_test"],
        y_pred_test=labelled_residual_split["y_pred_test"],
        y_true_labels=labelled_residual_split["labels"],
        n_trials=6,
        n_initial=3,
        random_state=0,
    )
    assert "ord" in best_params
    assert len(trials_df) == 6
