# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI dispatcher for the ``spotanomaly2 benchmark`` subcommand.

Two modes are sketched here:

- ``synthetic`` (default, fully implemented): generates a deterministic
  AR(1) "true" series, adds small Gaussian noise to simulate a forecaster,
  injects point/contextual/collective anomalies into the test split, and
  runs every scorer through the pipeline. This produces a real benchmark
  with controlled labels — the exact setup needed to back the
  recommendation with statistics rather than vibes.

- ``real`` (stub, raises): use processed panel data + existing trained
  forecaster models. Wiring this to ``Pipeline.detect()`` is a follow-up.

The orchestration here is intentionally self-contained — it does not call
``Pipeline``, so the benchmark can run before the user has trained any
forecaster model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spotanomaly2.infrastructure.storage import generate_timestamp
from spotanomaly2.research import scorer_runner
from spotanomaly2.research.labels import labels_from_known, merge_labels
from spotanomaly2.research.report import render
from spotanomaly2.research.scorer_tuner import tune_scorer
from spotanomaly2.research.synthetic import inject_all

_DEFAULT_SCORERS = ("KMeansScorer", "NormScorer", "IsolationForestScorer", "GMMScorer")


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------


def add_benchmark_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Compare and tune anomaly scorers on synthetic or real data",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=("synthetic", "real"),
        default="synthetic",
        help="Data source. 'synthetic' generates AR(1) data + controlled "
        "anomalies (default). 'real' uses processed panels + existing "
        "models (not yet wired).",
    )
    parser.add_argument(
        "--scorers",
        nargs="*",
        default=list(_DEFAULT_SCORERS),
        help="Scorers to benchmark (default: all four).",
    )
    parser.add_argument(
        "--tune",
        dest="tune",
        action="store_true",
        default=True,
        help="Run Bayesian tuning per scorer (default).",
    )
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Skip tuning; use defaults from each scorer's config search space.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of SpotOptim trials per scorer (default from config.benchmark).",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=None,
        help="Override SpotOptim initial design size (default from config.benchmark).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: data/benchmarks/<timestamp>).",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=2016,
        help="Synthetic mode: training window length (default 2016 = 7d at 5min).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=2016,
        help="Synthetic mode: test window length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed for synthetic data generation (default from config).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run the harness across multiple seeds and aggregate. Each seed "
        "produces an independent synthetic split; per-scorer metrics are "
        "averaged. Overrides --seed.",
    )
    parser.add_argument(
        "--forecaster-noise-scale",
        type=float,
        default=None,
        help="Synthetic mode: residual noise scale of the simulated forecaster. "
        "Larger = harder problem (anomalies must compete with bigger forecast "
        "errors). Default 0.6.",
    )


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def _ar1_series(n: int, rng: np.random.Generator, phi: float = 0.7, noise_scale: float = 1.0) -> np.ndarray:
    out = np.empty(n, dtype=float)
    out[0] = rng.normal(scale=noise_scale)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.normal(scale=noise_scale)
    return out


def _generate_synthetic_split(
    n_train: int,
    n_test: int,
    config: dict[str, Any],
    seed: int,
    forecaster_noise_scale: float = 0.6,
) -> dict[str, Any]:
    """Build train/test forecaster inputs + injected labels.

    The "true" series is AR(1) on two channels. The "forecast" is the true
    series with added Gaussian noise (so residuals are dominated by that
    noise plus any injected anomalies in test). Anomaly injection is
    pre-forecast on the *test* split only, per the design plan.
    """
    rng = np.random.default_rng(seed)
    freq = config.get("process", {}).get("resample", {}).get("freq", "5min")
    start = pd.Timestamp("2026-01-01", tz="UTC")
    full_idx = pd.date_range(start, periods=n_train + n_test, freq=freq, tz="UTC")

    n_features = 2
    cols = [f"channel_{i}" for i in range(n_features)]
    true_full = np.column_stack(
        [_ar1_series(n_train + n_test, rng, phi=0.7, noise_scale=1.0) for _ in cols]
    )
    y_true_full = pd.DataFrame(true_full, index=full_idx, columns=cols)

    forecast_noise = rng.normal(loc=0.0, scale=forecaster_noise_scale, size=true_full.shape)
    y_pred_full = pd.DataFrame(true_full + forecast_noise, index=full_idx, columns=cols)

    train_idx = full_idx[:n_train]
    test_idx = full_idx[n_train:]

    y_true_train = y_true_full.loc[train_idx]
    y_pred_train = y_pred_full.loc[train_idx]

    # Inject pre-forecast anomalies in test only (label assembly aligns
    # with this index).
    y_true_test_clean = y_true_full.loc[test_idx]
    inj_result = inject_all(y_true_test_clean, config=config, rng=rng, columns=cols)
    y_true_test = inj_result.df
    y_pred_test = y_pred_full.loc[test_idx]  # forecaster doesn't see injection

    # Combine known-anomaly labels (from config) and synthetic labels.
    known = config.get("known_anomalies") or []
    known_labels = labels_from_known(test_idx, known)
    labels = merge_labels(inj_result.labels, known_labels)

    return {
        "y_true_train": y_true_train,
        "y_pred_train": y_pred_train,
        "y_true_test": y_true_test,
        "y_pred_test": y_pred_test,
        "labels": labels.to_numpy(dtype=int),
        "test_index": test_idx,
        "injection_sites": inj_result.sites,
    }


# ---------------------------------------------------------------------------
# Top-level run() — wired into main.py
# ---------------------------------------------------------------------------


def _resolve_search_space(scorer_name: str, benchmark_cfg: dict[str, Any]) -> dict[str, Any]:
    spaces = benchmark_cfg.get("search_spaces", {}) or {}
    space = spaces.get(scorer_name)
    if space is None:
        raise KeyError(
            f"No search space defined for {scorer_name!r} under benchmark.search_spaces "
            f"in config; either add one or run with --no-tune."
        )
    # Search-space entries are stored as lists in YAML; coerce 2/3-tuples back.
    coerced: dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, list) and len(v) in (2, 3) and all(isinstance(x, (int, float)) for x in v[:2]):
            coerced[k] = tuple(v)
        else:
            coerced[k] = v
    return coerced


def _resolve_default_params(scorer_name: str, benchmark_cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull a sensible default param dict for a scorer when --no-tune is used."""
    defaults = (benchmark_cfg.get("defaults") or {}).get(scorer_name, {})
    return dict(defaults)


def _format_invocation(args: argparse.Namespace) -> str:
    parts = ["uv run spotanomaly2 benchmark"]
    if args.config:
        parts.append(f"--config {args.config}")
    parts.append(f"--mode {args.mode}")
    if args.scorers != list(_DEFAULT_SCORERS):
        parts.append(f"--scorers {' '.join(args.scorers)}")
    parts.append("--tune" if args.tune else "--no-tune")
    if args.n_trials:
        parts.append(f"--n-trials {args.n_trials}")
    if args.n_initial:
        parts.append(f"--n-initial {args.n_initial}")
    if args.output:
        parts.append(f"--output {args.output}")
    return " ".join(parts)


def _resolve_seeds(args: argparse.Namespace, benchmark_cfg: dict[str, Any]) -> list[int]:
    if args.seeds:
        return [int(s) for s in args.seeds]
    if args.seed is not None:
        return [int(args.seed)]
    cfg_seeds = benchmark_cfg.get("seeds")
    if cfg_seeds:
        return [int(s) for s in cfg_seeds]
    return [int(benchmark_cfg.get("random_state", 42))]


def _resolve_forecaster_noise(args: argparse.Namespace, benchmark_cfg: dict[str, Any]) -> float:
    if args.forecaster_noise_scale is not None:
        return float(args.forecaster_noise_scale)
    return float(benchmark_cfg.get("forecaster_noise_scale", 0.6))


def run(args: argparse.Namespace, config: dict[str, Any], logger: logging.Logger) -> int:
    if args.mode == "real":
        logger.error(
            "Benchmark mode='real' is not yet wired. Run with --mode synthetic, "
            "or hook this up to Pipeline.detect() in spotanomaly2/research/cli.py::run."
        )
        return 2

    benchmark_cfg = config.get("benchmark") or {}
    n_trials = args.n_trials or int(benchmark_cfg.get("n_trials", 30))
    n_initial = args.n_initial or int(benchmark_cfg.get("n_initial", 8))
    seeds = _resolve_seeds(args, benchmark_cfg)
    forecaster_noise_scale = _resolve_forecaster_noise(args, benchmark_cfg)
    high_quantile = float(benchmark_cfg.get("pinned_high_quantile", 0.995))

    timestamp = generate_timestamp()
    out_dir = args.output or Path(config.get("paths", {}).get("results_dir", "data/results")).parent / "benchmarks" / timestamp
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Benchmark output directory: %s", out_dir)
    logger.info("Seeds: %s | forecaster_noise_scale=%.2f | n_trials=%d n_initial=%d", seeds, forecaster_noise_scale, n_trials, n_initial)

    runs: list[dict[str, Any]] = []
    tuning_trials: dict[str, pd.DataFrame] = {}
    # Aggregate metrics keyed by (seed, scorer) so the report can average.
    per_seed_runs: list[dict[str, Any]] = []
    representative_split: dict[str, Any] | None = None

    for seed in seeds:
        logger.info("=== seed=%d ===", seed)
        split = _generate_synthetic_split(
            args.n_train, args.n_test, config, seed=seed, forecaster_noise_scale=forecaster_noise_scale
        )
        if representative_split is None:
            representative_split = split
        n_pos = int(split["labels"].sum())
        logger.info(
            "Test set (seed=%d): %d rows, %d injected anomalies (prevalence=%.4f)",
            seed, len(split["labels"]), n_pos, n_pos / max(len(split["labels"]), 1),
        )

        for scorer_name in args.scorers:
            logger.info("--- seed=%d scorer=%s ---", seed, scorer_name)
            if args.tune:
                try:
                    search_space = _resolve_search_space(scorer_name, benchmark_cfg)
                except KeyError as exc:
                    logger.error("Skipping %s: %s", scorer_name, exc)
                    continue
                logger.info(
                    "Tuning %s for %d trials (initial=%d) over %s",
                    scorer_name, n_trials, n_initial, search_space,
                )
                best_params, trials_df = tune_scorer(
                    scorer_name=scorer_name,
                    search_space=search_space,
                    y_true_train=split["y_true_train"],
                    y_pred_train=split["y_pred_train"],
                    y_true_test=split["y_true_test"],
                    y_pred_test=split["y_pred_test"],
                    y_true_labels=split["labels"],
                    n_trials=n_trials,
                    n_initial=n_initial,
                    high_quantile=high_quantile,
                    random_state=seed,
                )
                trials_df["seed"] = seed
                key = scorer_name
                if key in tuning_trials:
                    tuning_trials[key] = pd.concat([tuning_trials[key], trials_df], ignore_index=True)
                else:
                    tuning_trials[key] = trials_df
                logger.info(
                    "Best %s params (seed=%d): %s (PR-AUC max in trials: %.4f)",
                    scorer_name, seed, best_params, trials_df["pr_auc"].max(),
                )
                params_to_use = best_params
            else:
                params_to_use = _resolve_default_params(scorer_name, benchmark_cfg)
                logger.info("Using default params for %s: %s", scorer_name, params_to_use)

            try:
                metrics = scorer_runner.run_scorer(
                    scorer_name,
                    params_to_use,
                    y_true_train=split["y_true_train"],
                    y_pred_train=split["y_pred_train"],
                    y_true_test=split["y_true_test"],
                    y_pred_test=split["y_pred_test"],
                    y_true_labels=split["labels"],
                    high_quantile=high_quantile,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Scorer %s (seed=%d) failed at final run: %s", scorer_name, seed, exc, exc_info=True)
                continue

            metrics["seed"] = seed
            logger.info(
                "%s seed=%d | PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f fit=%.2fs",
                scorer_name, seed,
                metrics.get("pr_auc", float("nan")),
                metrics.get("roc_auc", float("nan")),
                metrics.get("f1", float("nan")),
                metrics.get("fit_seconds") or 0.0,
            )
            per_seed_runs.append(metrics)

    if not per_seed_runs:
        logger.error("No scorer runs succeeded; nothing to render.")
        return 1

    # The leaderboard averages metrics across seeds (one row per
    # (scorer, seed) so report.py's groupby(scorer).agg can collapse
    # them). For curve plots we keep the first seed's run only — overlaying
    # 4 scorers × N seeds gets unreadable fast.
    runs = [r for r in per_seed_runs if r.get("seed") == seeds[0]]
    if not runs:
        runs = per_seed_runs[: len(args.scorers)]
    split = representative_split if representative_split is not None else split  # type: ignore[has-type]

    metrics_df = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("raw_scores", "flags", "test_index")}
            for r in per_seed_runs
        ]
    )
    if len(seeds) > 1:
        agg = (
            metrics_df.groupby("scorer")
            .agg(
                pr_auc_mean=("pr_auc", "mean"),
                pr_auc_std=("pr_auc", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                n_seeds=("seed", "nunique"),
            )
            .round(4)
        )
        logger.info("Multi-seed aggregation:\n%s", agg.to_string())

    invocation = _format_invocation(args)
    paths = render(
        runs=runs,
        metrics_df=metrics_df,
        y_true=split["labels"],
        output_dir=out_dir,
        cli_invocation=invocation,
        tuning_trials=tuning_trials or None,
    )
    logger.info("Wrote benchmark artifacts:")
    for k, v in paths.items():
        logger.info("  %-18s %s", k, v)
    return 0


def benchmark_inputs_for_tests(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Test hook — exposes the synthetic split builder by name."""
    return _generate_synthetic_split(*args, **kwargs)
