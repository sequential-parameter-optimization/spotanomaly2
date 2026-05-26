# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotanomaly2.research.report (smoke-level)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.research.report import render


@pytest.fixture
def fake_runs() -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(0)
    n = 300
    y_true = (rng.random(n) < 0.05).astype(int)
    runs = []
    for name, signal in (("KMeansScorer", 0.6), ("NormScorer", 0.4), ("GMMScorer", 0.5)):
        scores = rng.random(n) + signal * y_true
        flags = (scores > scores.mean() + 1.5 * scores.std()).astype(int)
        runs.append(
            {
                "scorer": name,
                "params": {},
                "raw_scores": scores,
                "flags": flags,
                "pr_auc": float(0.2 + signal * 0.5),
                "roc_auc": float(0.5 + signal * 0.4),
                "average_precision": float(0.2 + signal * 0.5),
                "f1": 0.4,
                "fit_seconds": 0.2 + signal * 0.1,
                "score_seconds": 0.05,
                "n_total": n,
                "n_positive": int(y_true.sum()),
                "prevalence": float(y_true.mean()),
            }
        )
    return runs, y_true


def test_render_writes_all_artifacts(tmp_path: Path, fake_runs) -> None:
    runs, y_true = fake_runs
    metrics_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("raw_scores", "flags", "test_index")} for r in runs]
    )
    paths = render(runs, metrics_df, y_true, tmp_path, cli_invocation="uv run spotanomaly2 benchmark")
    assert paths["html"].exists()
    assert paths["md"].exists()
    assert paths["metrics"].exists()
    for k in ("leaderboard_png", "pr_png", "roc_png", "hist_png", "runtime_png"):
        assert paths[k].exists(), f"Missing {k}"
    md_text = paths["md"].read_text(encoding="utf-8")
    # Markdown should reference the relative figure paths.
    assert "figures/pr_curves.png" in md_text
    assert "PR-AUC" in md_text


def test_render_writes_tuning_trials(tmp_path: Path, fake_runs) -> None:
    runs, y_true = fake_runs
    metrics_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("raw_scores", "flags", "test_index")} for r in runs]
    )
    trials = {"KMeansScorer": pd.DataFrame({"trial_idx": [0, 1], "pr_auc": [0.4, 0.5]})}
    paths = render(runs, metrics_df, y_true, tmp_path, tuning_trials=trials)
    assert "tuning" in paths and paths["tuning"].exists()
