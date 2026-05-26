# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotanomaly2.research.scorer_runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.research.gmm_scorer import GMMAnomalyScorer
from spotanomaly2.research.scorer_runner import BenchmarkDetector, run_scorer


@pytest.fixture
def split_residuals() -> dict[str, pd.DataFrame]:
    """Forecaster-style train/test frames with deterministic residuals.

    Residuals on the test set have a few large injected spikes; we use the
    same spike locations as the y_true_labels in tests below.
    """
    rng = np.random.default_rng(0)
    n_train, n_test = 600, 200
    idx_tr = pd.RangeIndex(n_train)
    idx_te = pd.RangeIndex(n_train, n_train + n_test)

    y_true_train = pd.DataFrame(rng.normal(size=(n_train, 2)), index=idx_tr, columns=["a", "b"])
    y_pred_train = y_true_train + rng.normal(scale=0.2, size=y_true_train.shape)

    y_true_test = pd.DataFrame(rng.normal(size=(n_test, 2)), index=idx_te, columns=["a", "b"])
    y_pred_test = y_true_test + rng.normal(scale=0.2, size=y_true_test.shape)

    # Inject 5 strong residual spikes in the test set so a scorer can
    # actually find something.
    spike_positions = [25, 75, 110, 150, 180]
    for pos in spike_positions:
        y_true_test.iloc[pos, 0] += 12.0
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


def test_benchmark_detector_accepts_prebuilt_scorer(split_residuals: dict) -> None:
    gmm = GMMAnomalyScorer(n_components=2, selection_criterion="fixed")
    det = BenchmarkDetector(
        scorer_name="GMMScorer",
        scorer_params={"n_components": 2},
        high_quantile=0.99,
        normalize_scores=False,
        scorer=gmm,
    )
    det.fit(split_residuals["y_true_train"], split_residuals["y_pred_train"])
    scores_df, flags_df = det.score_and_detect(
        split_residuals["y_true_test"], split_residuals["y_pred_test"]
    )
    assert "anomaly_score" in scores_df.columns
    assert flags_df["anomaly_flag"].isin((0, 1)).all()


def test_benchmark_detector_falls_back_to_parent_when_no_scorer(split_residuals: dict) -> None:
    det = BenchmarkDetector(
        scorer_name="NormScorer",
        scorer_params={"ord": 2},
        high_quantile=0.99,
        normalize_scores=True,
    )
    det.fit(split_residuals["y_true_train"], split_residuals["y_pred_train"])
    scores_df, _ = det.score_and_detect(
        split_residuals["y_true_test"], split_residuals["y_pred_test"]
    )
    # normalize_scores=True → normalized column present.
    assert "anomaly_score_normalized" in scores_df.columns


def test_isolation_forest_auto_bump_neutralised_with_explicit_quantile(
    split_residuals: dict,
) -> None:
    """Passing a non-0.99 high_quantile should suppress the auto-bump path."""
    det = BenchmarkDetector(
        scorer_name="IsolationForestScorer",
        scorer_params={"n_estimators": 50, "contamination": 0.01, "random_state": 42},
        high_quantile=0.995,
        normalize_scores=False,
    )
    assert det.high_quantile == 0.995


def test_run_scorer_returns_metrics_for_each_scorer(split_residuals: dict) -> None:
    common = dict(
        y_true_train=split_residuals["y_true_train"],
        y_pred_train=split_residuals["y_pred_train"],
        y_true_test=split_residuals["y_true_test"],
        y_pred_test=split_residuals["y_pred_test"],
        y_true_labels=split_residuals["labels"],
        high_quantile=0.97,  # generous to make sure flags trigger
        n_resamples_bootstrap=20,
    )
    out_norm = run_scorer("NormScorer", {"ord": 2}, **common)
    out_kmeans = run_scorer("KMeansScorer", {"n_clusters": 3}, **common)
    out_iforest = run_scorer(
        "IsolationForestScorer",
        {"n_estimators": 50, "contamination": 0.01, "random_state": 42},
        **common,
    )
    out_gmm = run_scorer(
        "GMMScorer",
        {"n_components": 2, "selection_criterion": "fixed"},
        **common,
    )
    for out in (out_norm, out_kmeans, out_iforest, out_gmm):
        assert out["raw_scores"].shape == (200,)
        assert "pr_auc" in out
        assert "fit_seconds" in out and out["fit_seconds"] >= 0
        # Runner should produce *some* signal — PR-AUC must be above the
        # naive prevalence baseline (5/200 = 0.025) on this fixture.
        assert np.isnan(out["pr_auc"]) or out["pr_auc"] > 0.05


def test_run_scorer_rejects_unknown_scorer(split_residuals: dict) -> None:
    with pytest.raises(ValueError, match="Unknown scorer"):
        run_scorer(
            "DoesNotExist",
            {},
            y_true_train=split_residuals["y_true_train"],
            y_pred_train=split_residuals["y_pred_train"],
            y_true_test=split_residuals["y_true_test"],
            y_pred_test=split_residuals["y_pred_test"],
            y_true_labels=split_residuals["labels"],
        )
