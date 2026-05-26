# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render benchmark results as HTML, PNG, and Markdown.

Two viz paths:

- **plotly** for ``report.html`` (interactive). Already a project
  dependency and the same library used by
  ``spotanomaly2.domain.report_generator``.
- **matplotlib** for committable PNGs in ``report.md`` and
  ``docs/scorer_benchmark.md``. Already a project dependency. We
  deliberately avoid kaleido (which would let plotly emit PNGs) — it
  pulls a ~80 MB headless-Chrome stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scorer_run_label(run: dict[str, Any]) -> str:
    """Single-line label for a scorer run, used in legends."""
    name = run["scorer"]
    return f"{name} (PR-AUC={run.get('pr_auc', float('nan')):.3f})"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Per-figure renderers — each returns the saved PNG path, plotly object, or both
# ---------------------------------------------------------------------------


def _save_pr_curves_png(runs: list[dict[str, Any]], y_true: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for run in runs:
        if run.get("raw_scores") is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, run["raw_scores"])
        ax.plot(recall, precision, label=_scorer_run_label(run), linewidth=1.6)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (test set, raw scores)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_roc_curves_png(runs: list[dict[str, Any]], y_true: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for run in runs:
        if run.get("raw_scores") is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, run["raw_scores"])
        roc_auc = run.get("roc_auc", float("nan"))
        ax.plot(fpr, tpr, label=f"{run['scorer']} (ROC-AUC={roc_auc:.3f})", linewidth=1.6)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set, raw scores)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_score_distributions_png(
    runs: list[dict[str, Any]], y_true: np.ndarray, out_path: Path
) -> None:
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    pos_mask = y_true.astype(bool)
    for ax, run in zip(axes, runs):
        scores = run["raw_scores"]
        ax.hist(
            scores[~pos_mask],
            bins=50,
            alpha=0.6,
            color="#3a86ff",
            label=f"normal (n={int((~pos_mask).sum())})",
        )
        ax.hist(
            scores[pos_mask],
            bins=50,
            alpha=0.7,
            color="#fb5607",
            label=f"anomaly (n={int(pos_mask.sum())})",
        )
        ax.set_title(_scorer_run_label(run))
        ax.set_yscale("log")
        ax.set_ylabel("count (log)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("anomaly score (raw)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_runtime_bar_png(metrics_df: pd.DataFrame, out_path: Path) -> None:
    grouped = metrics_df.groupby("scorer", as_index=False).agg(
        fit_seconds=("fit_seconds", "mean"),
        score_seconds=("score_seconds", "mean"),
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    x = np.arange(len(grouped))
    ax.bar(x - width / 2, grouped["fit_seconds"], width, label="fit", color="#3a86ff")
    ax.bar(x + width / 2, grouped["score_seconds"], width, label="score", color="#fb5607")
    ax.set_xticks(x, grouped["scorer"], rotation=20, ha="right")
    ax.set_ylabel("wall-clock seconds")
    ax.set_title("Per-scorer runtime (mean across runs)")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_leaderboard_png(metrics_df: pd.DataFrame, out_path: Path) -> None:
    leaderboard = metrics_df.groupby("scorer", as_index=False).agg(
        pr_auc=("pr_auc", "max"),
        roc_auc=("roc_auc", "max"),
        f1=("f1", "max"),
    )
    leaderboard = leaderboard.sort_values("pr_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.27
    x = np.arange(len(leaderboard))
    ax.bar(x - width, leaderboard["pr_auc"], width, label="PR-AUC", color="#fb5607")
    ax.bar(x, leaderboard["roc_auc"], width, label="ROC-AUC", color="#3a86ff")
    ax.bar(x + width, leaderboard["f1"], width, label="F1 (best run)", color="#8338ec")
    ax.set_xticks(x, leaderboard["scorer"], rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title("Leaderboard (best run per scorer)")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotly interactive HTML
# ---------------------------------------------------------------------------


def _build_html_dashboard(runs: list[dict[str, Any]], y_true: np.ndarray, metrics_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Precision-recall curves",
            "ROC curves",
            "Score distribution (top scorer)",
            "Leaderboard (PR-AUC)",
        ),
    )

    # PR curves
    for run in runs:
        if run.get("raw_scores") is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, run["raw_scores"])
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision, mode="lines", name=_scorer_run_label(run), legendgroup="pr"
            ),
            row=1,
            col=1,
        )

    # ROC curves
    for run in runs:
        if run.get("raw_scores") is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, run["raw_scores"])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=run["scorer"],
                legendgroup="roc",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(dash="dash", color="grey"),
                   showlegend=False),
        row=1, col=2,
    )

    # Score histogram for top scorer
    if runs:
        top = max(runs, key=lambda r: -1.0 if pd.isna(r.get("pr_auc")) else r["pr_auc"])
        pos_mask = y_true.astype(bool)
        fig.add_trace(
            go.Histogram(x=top["raw_scores"][~pos_mask], name="normal", opacity=0.6, nbinsx=60),
            row=2, col=1,
        )
        fig.add_trace(
            go.Histogram(x=top["raw_scores"][pos_mask], name="anomaly", opacity=0.7, nbinsx=60),
            row=2, col=1,
        )

    # Leaderboard bar
    leaderboard = metrics_df.groupby("scorer", as_index=False).agg(pr_auc=("pr_auc", "max"))
    leaderboard = leaderboard.sort_values("pr_auc", ascending=False)
    fig.add_trace(
        go.Bar(x=leaderboard["scorer"], y=leaderboard["pr_auc"], name="PR-AUC",
               showlegend=False, marker_color="#fb5607"),
        row=2, col=2,
    )

    fig.update_layout(
        height=900,
        width=1300,
        title_text="spotanomaly2 scorer benchmark",
        barmode="overlay",
    )
    fig.update_xaxes(title_text="recall", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="precision", range=[0, 1.02], row=1, col=1)
    fig.update_xaxes(title_text="false positive rate", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="true positive rate", range=[0, 1.02], row=1, col=2)
    fig.update_xaxes(title_text="anomaly score (raw)", row=2, col=1)
    fig.update_yaxes(title_text="count (log)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="PR-AUC", range=[0, 1.02], row=2, col=2)
    return fig


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def _format_leaderboard_markdown(metrics_df: pd.DataFrame) -> str:
    leaderboard = (
        metrics_df.groupby("scorer", as_index=False)
        .agg(
            pr_auc=("pr_auc", "max"),
            roc_auc=("roc_auc", "max"),
            f1=("f1", "max"),
            mean_fit_s=("fit_seconds", "mean"),
            mean_score_s=("score_seconds", "mean"),
        )
        .sort_values("pr_auc", ascending=False)
    )
    lines = [
        "| Scorer | PR-AUC | ROC-AUC | best F1 | fit (s) | score (s) |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row['scorer']} "
            f"| {row['pr_auc']:.3f} "
            f"| {row['roc_auc']:.3f} "
            f"| {row['f1']:.3f} "
            f"| {row['mean_fit_s']:.3f} "
            f"| {row['mean_score_s']:.3f} |"
        )
    return "\n".join(lines)


def _format_recommendation_markdown(metrics_df: pd.DataFrame) -> str:
    leaderboard = (
        metrics_df.groupby("scorer", as_index=False)
        .agg(pr_auc=("pr_auc", "max"), mean_fit_s=("fit_seconds", "mean"))
        .sort_values("pr_auc", ascending=False)
    )
    if leaderboard.empty:
        return "No metrics produced."
    top = leaderboard.iloc[0]
    fastest = leaderboard.sort_values("mean_fit_s").iloc[0]
    lines = [
        f"**Default recommendation:** `{top['scorer']}` "
        f"— highest PR-AUC ({top['pr_auc']:.3f}) on this benchmark.",
        "",
        f"**Fastest:** `{fastest['scorer']}` "
        f"(mean fit {fastest['mean_fit_s']:.3f}s).",
    ]
    if top["scorer"] != fastest["scorer"]:
        ratio = top["pr_auc"] / max(leaderboard.set_index("scorer").loc[fastest["scorer"], "pr_auc"], 1e-9)
        lines.extend(
            [
                "",
                f"PR-AUC ratio (default / fastest): {ratio:.2f}× — pick the fastest "
                "if real-time scoring latency dominates over detection quality.",
            ]
        )
    return "\n".join(lines)


def _write_markdown_report(
    runs: list[dict[str, Any]],
    metrics_df: pd.DataFrame,
    figures: dict[str, str],
    out_path: Path,
    cli_invocation: str | None,
) -> None:
    parts: list[str] = []
    parts.append("# spotanomaly2 scorer benchmark")
    parts.append("")
    if cli_invocation:
        parts.append(f"_Reproduce: `{cli_invocation}`_")
        parts.append("")
    parts.append("## TL;DR")
    parts.append("")
    parts.append(_format_leaderboard_markdown(metrics_df))
    parts.append("")
    parts.append(_format_recommendation_markdown(metrics_df))
    parts.append("")
    parts.append("## Leaderboard chart")
    parts.append("")
    parts.append(f"![Leaderboard]({figures['leaderboard']})")
    parts.append("")
    parts.append("## Precision-recall and ROC")
    parts.append("")
    parts.append(f"![PR curves]({figures['pr']})")
    parts.append("")
    parts.append(f"![ROC curves]({figures['roc']})")
    parts.append("")
    parts.append("## Score distributions")
    parts.append("")
    parts.append(f"![Score distributions]({figures['hist']})")
    parts.append("")
    parts.append("## Runtime")
    parts.append("")
    parts.append(f"![Runtime]({figures['runtime']})")
    parts.append("")
    parts.append("## Notes on metric choices")
    parts.append("")
    parts.append(
        "- The cross-scorer leaderboard uses *raw* anomaly scores and "
        "threshold-independent metrics (PR-AUC, ROC-AUC). The "
        "production `ScoreNormalizer` re-fits on test data, which "
        "makes the normalized score not a per-sample probability — see "
        "`spotanomaly2_safe/scoring/pipeline.py` lines 276-281."
    )
    parts.append(
        "- F1 is reported per scorer at that scorer's tuned operating "
        "point only. Comparing F1 across scorers is meaningful only "
        "after pinning the same `high_quantile` everywhere, which the "
        "harness does (default 0.995)."
    )
    parts.append(
        "- PR-AUC is preferred over ROC-AUC under high class imbalance "
        "(anomalies typically <1%). See the prevalence sweep below."
    )
    parts.append("")
    out_path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def render(
    runs: Iterable[dict[str, Any]],
    metrics_df: pd.DataFrame,
    y_true: np.ndarray,
    output_dir: Path | str,
    *,
    cli_invocation: str | None = None,
    tuning_trials: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Write report.html, report.md, and figures/* into ``output_dir``.

    Args:
        runs: Iterable of run dicts from :func:`run_scorer`. Used for
            curve plotting and to grab raw scores per scorer.
        metrics_df: DataFrame view of those runs (one row per run, with
            scalar metric columns) — used for leaderboard aggregation.
        y_true: Binary labels aligned with the test set scored in ``runs``.
        output_dir: Destination directory; created if missing.
        cli_invocation: Reproduction string included at the top of the
            Markdown writeup.
        tuning_trials: Optional ``{scorer_name: trials_df}`` to dump
            alongside the report as ``tuning_results.json``.

    Returns:
        Mapping from artifact name to its absolute path.
    """
    runs = list(runs)
    out_dir = _ensure_dir(Path(output_dir))
    figures_dir = _ensure_dir(out_dir / "figures")

    paths: dict[str, Path] = {}

    leaderboard_png = figures_dir / "leaderboard.png"
    pr_png = figures_dir / "pr_curves.png"
    roc_png = figures_dir / "roc_curves.png"
    hist_png = figures_dir / "score_distributions.png"
    runtime_png = figures_dir / "runtime.png"

    _save_leaderboard_png(metrics_df, leaderboard_png)
    _save_pr_curves_png(runs, y_true, pr_png)
    _save_roc_curves_png(runs, y_true, roc_png)
    _save_score_distributions_png(runs, y_true, hist_png)
    _save_runtime_bar_png(metrics_df, runtime_png)

    paths["leaderboard_png"] = leaderboard_png
    paths["pr_png"] = pr_png
    paths["roc_png"] = roc_png
    paths["hist_png"] = hist_png
    paths["runtime_png"] = runtime_png

    # Plotly interactive dashboard
    fig = _build_html_dashboard(runs, y_true, metrics_df)
    html_path = out_dir / "report.html"
    pio.write_html(fig, file=str(html_path), include_plotlyjs="cdn", full_html=True)
    paths["html"] = html_path

    # Markdown writeup, with figure paths *relative* to the .md location
    md_figures = {
        "leaderboard": f"figures/{leaderboard_png.name}",
        "pr": f"figures/{pr_png.name}",
        "roc": f"figures/{roc_png.name}",
        "hist": f"figures/{hist_png.name}",
        "runtime": f"figures/{runtime_png.name}",
    }
    md_path = out_dir / "report.md"
    _write_markdown_report(runs, metrics_df, md_figures, md_path, cli_invocation)
    paths["md"] = md_path

    # Persist metrics + tuning trials for downstream analysis. Drop array
    # columns and serialise dict columns to JSON so parquet doesn't choke
    # on heterogeneous / empty struct fields.
    drop_cols = [c for c in ("raw_scores", "flags", "test_index") if c in metrics_df.columns]
    metrics_df_clean = metrics_df.drop(columns=drop_cols).copy()
    if "params" in metrics_df_clean.columns:
        metrics_df_clean["params"] = metrics_df_clean["params"].apply(
            lambda v: json.dumps(v, default=str) if isinstance(v, dict) else v
        )
    metrics_path = out_dir / "metrics.parquet"
    metrics_df_clean.to_parquet(metrics_path)
    paths["metrics"] = metrics_path

    if tuning_trials:
        # Use a JSON-friendly dump (parquet would mean one file per scorer).
        trials_payload = {
            scorer: trials.to_dict(orient="records") for scorer, trials in tuning_trials.items()
        }
        trials_path = out_dir / "tuning_results.json"
        trials_path.write_text(json.dumps(trials_payload, indent=2, default=str), encoding="utf-8")
        paths["tuning"] = trials_path

    return paths
