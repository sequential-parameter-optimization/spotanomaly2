# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run a configured scorer through the production detection pipeline.

Two pieces:

1. :class:`BenchmarkDetector` — a tiny subclass of
   :class:`spotanomaly2_safe.scoring.pipeline.ForecastingAnomalyDetector`
   that accepts a *pre-built* scorer object via the ``scorer=`` kwarg.
   This lets us inject scorers that don't live in ``spotanomaly2-safe``
   (notably :class:`GMMAnomalyScorer`) without monkey-patching the safe
   library or duplicating its fit/score/normalize/detect plumbing.

2. :func:`run_scorer` — single-shot wrapper that builds a detector, fits it
   on training residuals, scores the test set, computes metrics, and
   returns everything in a flat dict suitable for one row of a results
   DataFrame.

Apples-to-apples comparison rule: the caller passes a *pinned*
``high_quantile`` so the IsolationForest auto-bump
(``ForecastingAnomalyDetector.__init__`` lines 116-119) does not silently
shift one scorer's operating point relative to the others.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from spotanomaly2_safe.scoring.detector import QuantileAnomalyDetector
from spotanomaly2_safe.scoring.normalizer import ScoreNormalizer
from spotanomaly2_safe.scoring.pipeline import ForecastingAnomalyDetector

from spotanomaly2.research.gmm_scorer import GMMAnomalyScorer
from spotanomaly2.research.metrics import compute_metrics


class BenchmarkDetector(ForecastingAnomalyDetector):
    """ForecastingAnomalyDetector that can take a pre-built scorer object.

    Behaves identically to the parent when ``scorer`` is not supplied.
    When ``scorer`` is supplied, the parent's name-based if/elif tree and
    parameter whitelist are bypassed entirely; the caller is responsible
    for constructing the scorer with whatever parameters are appropriate.

    Critically, the IsolationForest-specific auto-quantile bump in the
    parent (``high_quantile=0.99 → 0.995``) is also bypassed when the
    caller supplies a scorer, so the benchmark can pin one quantile across
    every scorer for a fair comparison.
    """

    def __init__(
        self,
        scorer_name: str = "KMeansScorer",
        scorer_params: Optional[dict[str, Any]] = None,
        high_quantile: float = 0.99,
        normalize_scores: bool = True,
        normalization_quantile: float = 0.99,
        scorer: Any = None,
    ) -> None:
        if scorer is None:
            super().__init__(
                scorer_name=scorer_name,
                scorer_params=scorer_params,
                high_quantile=high_quantile,
                normalize_scores=normalize_scores,
                normalization_quantile=normalization_quantile,
            )
            return

        # Inline the parent's initialisation, but skip the name-based
        # scorer construction and the IsolationForest quantile bump.
        if not (hasattr(scorer, "fit") and hasattr(scorer, "score")):
            raise TypeError(
                "scorer must expose fit(residuals) and score(residuals) methods"
            )

        self.scorer_name = scorer_name
        self.scorer_params = scorer_params or {}
        self.high_quantile = high_quantile
        self.normalize_scores = normalize_scores
        self.normalization_quantile = normalization_quantile

        self.scorer = scorer
        self.detector = QuantileAnomalyDetector(high_quantile=high_quantile)
        self.normalizer: Optional[ScoreNormalizer] = (
            ScoreNormalizer(normalization_quantile=normalization_quantile)
            if normalize_scores
            else None
        )


# ---------------------------------------------------------------------------
# Single-scorer benchmark runner
# ---------------------------------------------------------------------------


_BUILTIN_SCORER_NAMES = {"KMeansScorer", "NormScorer", "IsolationForestScorer"}
_GMM_SCORER_NAME = "GMMScorer"


def _build_detector(
    scorer_name: str,
    params: dict[str, Any],
    high_quantile: float,
) -> BenchmarkDetector:
    """Construct the detector for ``scorer_name`` with ``params``.

    GMM goes through the ``scorer=`` injection path; the three
    `spotanomaly2_safe` scorers go through the parent's name-based path.
    """
    if scorer_name == _GMM_SCORER_NAME:
        gmm = GMMAnomalyScorer(**params)
        return BenchmarkDetector(
            scorer_name=_GMM_SCORER_NAME,
            scorer_params=params,
            high_quantile=high_quantile,
            normalize_scores=False,  # we use raw scores for cross-scorer comparison
            scorer=gmm,
        )

    if scorer_name in _BUILTIN_SCORER_NAMES:
        return BenchmarkDetector(
            scorer_name=scorer_name,
            scorer_params=params,
            high_quantile=high_quantile,
            normalize_scores=False,
        )

    raise ValueError(
        f"Unknown scorer {scorer_name!r}; expected one of {sorted(_BUILTIN_SCORER_NAMES | {_GMM_SCORER_NAME})}"
    )


def run_scorer(
    scorer_name: str,
    params: dict[str, Any],
    y_true_train: pd.DataFrame,
    y_pred_train: pd.DataFrame,
    y_true_test: pd.DataFrame,
    y_pred_test: pd.DataFrame,
    y_true_labels: np.ndarray,
    *,
    high_quantile: float = 0.995,
    n_resamples_bootstrap: int = 200,
) -> dict[str, Any]:
    """Fit/score one scorer configuration and return its metrics.

    Args:
        scorer_name: One of ``"KMeansScorer"``, ``"NormScorer"``,
            ``"IsolationForestScorer"``, ``"GMMScorer"``.
        params: Scorer constructor kwargs.
        y_true_train, y_pred_train: Forecaster train inputs/outputs.
        y_true_test, y_pred_test: Forecaster test inputs/outputs.
        y_true_labels: Binary anomaly labels aligned with ``y_true_test.index``.
        high_quantile: Pinned operating threshold (default 0.995 — same for
            every scorer to neutralize the IsolationForest auto-bump).
        n_resamples_bootstrap: F1 CI bootstrap iterations.

    Returns:
        Flat dict suitable as one row of the leaderboard. Includes
        ``scorer``, ``params`` (JSON-serialisable repr), all metrics from
        :func:`compute_metrics`, and ``raw_scores`` / ``flags`` arrays for
        the report layer.
    """
    detector = _build_detector(scorer_name, params, high_quantile=high_quantile)

    fit_start = time.perf_counter()
    detector.fit(y_true_train, y_pred_train)
    fit_seconds = time.perf_counter() - fit_start

    score_start = time.perf_counter()
    scores_df, flags_df = detector.score_and_detect(y_true_test, y_pred_test)
    score_seconds = time.perf_counter() - score_start

    raw_scores = scores_df["anomaly_score"].to_numpy(dtype=float)
    flags = flags_df["anomaly_flag"].to_numpy(dtype=int)

    metrics = compute_metrics(
        y_true=np.asarray(y_true_labels, dtype=int),
        scores=raw_scores,
        flags=flags,
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        n_resamples_bootstrap=n_resamples_bootstrap,
    )

    return {
        "scorer": scorer_name,
        "params": dict(params),
        **metrics,
        "raw_scores": raw_scores,
        "flags": flags,
        "test_index": scores_df.index,
    }
