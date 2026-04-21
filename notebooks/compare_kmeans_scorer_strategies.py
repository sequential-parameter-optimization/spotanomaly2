"""Compare KMeans scorer strategies with optional train-tail trimming.

Why this script:
- You suspect anomalies may already be present in scorer training data.
- This script compares baseline KMeans scorer vs. trimmed variants where
  high-error rows are dropped before fitting the scorer.

Leakage safety:
- Uses AnomalyDetector's leakage guard to exclude forecaster-training rows.
- Splits only unseen data into scorer-train/scorer-test.

Usage:
    uv run python notebooks/compare_kmeans_scorer_strategies.py \
    --config config/config.yaml \
      --trim-quantiles 0.99,0.995,0.999 \
      --k-values 2,3,4,5 \
      --window-values 1,3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import re

import numpy as np
import pandas as pd

from spotanomaly2.application.config import load_config
from spotanomaly2.application.data_manager import DataManager
from spotanomaly2.domain.anomaly_detector import AnomalyDetector
from spotanomaly2.domain.spotforecast_adapter import SpotforecastTrainer
from spotanomaly2.infrastructure import logging
from spotanomaly2.infrastructure.storage import generate_timestamp
from spotanomaly2_safe.scoring.pipeline import ForecastingAnomalyDetector


@dataclass
class PanelMatrices:
    panel_id: str
    train_true: pd.DataFrame
    train_pred: pd.DataFrame
    test_true: pd.DataFrame
    test_pred: pd.DataFrame


@dataclass
class StrategyResult:
    strategy: str
    k: int
    window: int
    trim_quantile: Optional[float]
    anomaly_flags: int
    anomaly_rate: float
    scores_df: pd.DataFrame
    flags_df: pd.DataFrame


def _parse_csv_numbers(value: str, cast=float) -> list:
    if not value.strip():
        return []
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def _get_panel_matrices(
    panel_id: str,
    df: pd.DataFrame,
    cfg: dict,
    detector: AnomalyDetector,
) -> PanelMatrices:
    model_data = detector.load_forecasting_model(panel_id)
    history_df, unseen_df = detector._split_unseen_scoring_data(panel_id, df, model_data)

    hist_window = cfg["detect"]["hist_window"]
    target_date = cfg["detect"].get("target_date")

    if target_date:
        test_start_idx, test_end_idx = detector._calculate_test_window_with_target_date(
            unseen_df, target_date, hist_window
        )
    else:
        test_start_idx, test_end_idx = detector._calculate_test_window_default(
            unseen_df, hist_window
        )

    test_start_idx, test_end_idx = detector._adjust_window_for_insufficient_data(
        unseen_df, test_start_idx, test_end_idx, hist_window
    )

    train_df = unseen_df.iloc[:test_start_idx]
    test_df = unseen_df.iloc[test_start_idx:test_end_idx]

    adapter = SpotforecastTrainer(cfg, detector.logger)
    history_for_train_pred = history_df if len(history_df) > 0 else None
    train_pred_df = adapter.predict(model_data, train_df, history_df=history_for_train_pred)

    test_history_df = pd.concat([history_df, train_df]) if len(history_df) > 0 else train_df
    test_pred_df = adapter.predict(model_data, test_df, history_df=test_history_df)

    target_cols = train_pred_df.columns
    train_true_df = train_df.loc[train_pred_df.index, target_cols]
    test_true_df = test_df.loc[test_pred_df.index, target_cols]

    train_true_df, train_keep_mask = detector._exclude_imputed_rows(
        panel_id=panel_id,
        window_name="train",
        source_df=train_df,
        aligned_true_df=train_true_df,
    )
    train_pred_df = train_pred_df.loc[train_keep_mask]

    test_true_df, test_keep_mask = detector._exclude_imputed_rows(
        panel_id=panel_id,
        window_name="test",
        source_df=test_df,
        aligned_true_df=test_true_df,
    )
    test_pred_df = test_pred_df.loc[test_keep_mask]

    train_true_df, train_pred_df, test_true_df, test_pred_df = detector._exclude_invalid_scorer_fit_rows(
        panel_id=panel_id,
        fit_true_df=train_true_df,
        fit_pred_df=train_pred_df,
        eval_true_df=test_true_df,
        eval_pred_df=test_pred_df,
    )

    detector._validate_scoring_inputs(panel_id, test_true_df, test_pred_df)

    return PanelMatrices(
        panel_id=panel_id,
        train_true=train_true_df,
        train_pred=train_pred_df,
        test_true=test_true_df,
        test_pred=test_pred_df,
    )


def _iter_strategies(
    k_values: Iterable[int],
    window_values: Iterable[int],
    trim_quantiles: Iterable[float],
):
    for k in k_values:
        for w in window_values:
            yield {"name": f"k{k}_w{w}_baseline", "k": k, "window": w, "trim_q": None}
            for q in trim_quantiles:
                yield {"name": f"k{k}_w{w}_trim_q{q}", "k": k, "window": w, "trim_q": q}


def _known_anomaly_intervals(cfg: dict) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for item in cfg.get("known_anomalies", []):
        start = item.get("start")
        end = item.get("end")
        if not start or not end:
            continue
        intervals.append((pd.Timestamp(start), pd.Timestamp(end)))
    return intervals


def _build_known_anomaly_mask(index: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    mask = pd.Series(False, index=index)
    for start, end in _known_anomaly_intervals(cfg):
        mask |= (index >= start) & (index <= end)
    return mask


def _normalize_for_plot(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = float(s.std(skipna=True))
    if np.isnan(std) or std == 0.0:
        return pd.Series(0.0, index=s.index)
    mean = float(s.mean(skipna=True))
    return (s - mean) / std


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")


def _save_panel_detail_csv(
    panel_id: str,
    mats: PanelMatrices,
    strategy_results: list[StrategyResult],
    output_dir: Path,
    ts: str,
) -> Path:
    detail_df = pd.DataFrame(index=mats.test_true.index)

    for col in mats.test_true.columns:
        detail_df[f"true__{col}"] = mats.test_true[col]
    for col in mats.test_pred.columns:
        detail_df[f"pred__{col}"] = mats.test_pred[col]
    common_cols = [c for c in mats.test_true.columns if c in mats.test_pred.columns]
    for col in common_cols:
        detail_df[f"abs_err__{col}"] = (mats.test_true[col] - mats.test_pred[col]).abs()

    if common_cols:
        detail_df["mean_abs_err"] = (mats.test_true[common_cols] - mats.test_pred[common_cols]).abs().mean(axis=1)

    for res in strategy_results:
        key = _sanitize_name(res.strategy)
        detail_df[f"score__{key}"] = res.scores_df["anomaly_score"].reindex(detail_df.index)
        detail_df[f"score_norm__{key}"] = res.scores_df["anomaly_score_normalized"].reindex(detail_df.index)
        detail_df[f"flag__{key}"] = res.flags_df["anomaly_flag"].reindex(detail_df.index).fillna(0).astype(int)

    detail_df = detail_df.reset_index().rename(columns={"index": "timestamp"})

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"kmeans_strategy_detail_panel_{panel_id}_{ts}.csv"
    detail_df.to_csv(out_csv, index=False)
    return out_csv


def _save_panel_plot(
    panel_id: str,
    mats: PanelMatrices,
    strategy_results: list[StrategyResult],
    cfg: dict,
    output_dir: Path,
    ts: str,
    top_n: int,
    context_columns: Optional[list[str]],
    context_max_cols: int,
) -> Path:
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError(
            "plotly is required for visualization. Install it in your env."
        ) from exc

    if not strategy_results:
        raise RuntimeError(f"No strategy results to plot for panel {panel_id}")

    ranked = sorted(strategy_results, key=lambda x: (x.anomaly_flags, x.anomaly_rate))
    selected = ranked[: max(1, top_n)]

    residual_mean = (mats.test_true - mats.test_pred).abs().mean(axis=1)
    known_intervals = _known_anomaly_intervals(cfg)

    available_context_cols = [c for c in mats.test_true.columns if c in mats.test_pred.columns]
    if context_columns:
        selected_context_cols = [c for c in context_columns if c in available_context_cols]
    else:
        selected_context_cols = available_context_cols[: max(1, context_max_cols)]
    if not selected_context_cols:
        selected_context_cols = available_context_cols[:1]

    fig = make_subplots(
        rows=len(selected) + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[
            "Actual panel values (z-normalized) + mean absolute residual"
        ] + [
            f"{r.strategy} | flags={r.anomaly_flags} | rate={r.anomaly_rate:.4f}"
            for r in selected
        ],
    )

    for col in selected_context_cols:
        fig.add_trace(
            go.Scatter(
                x=mats.test_true.index,
                y=_normalize_for_plot(mats.test_true[col]).values,
                mode="lines",
                line={"width": 1},
                name=f"true: {col}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    residual_mean_full = (mats.test_true - mats.test_pred).abs().mean(axis=1)
    fig.add_trace(
        go.Scatter(
            x=residual_mean_full.index,
            y=_normalize_for_plot(residual_mean_full).values,
            mode="lines",
            line={"width": 2, "dash": "dot", "color": "black"},
            name="mean |y_true-y_pred|",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    for start, end in known_intervals:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="orange",
            opacity=0.10,
            line_width=0,
            row=1,
            col=1,
        )

    for i, res in enumerate(selected, start=2):
        score_series = res.scores_df["anomaly_score"]
        flags_series = res.flags_df["anomaly_flag"]
        flagged_idx = flags_series.index[flags_series > 0]

        fig.add_trace(
            go.Scatter(
                x=score_series.index,
                y=score_series.values,
                mode="lines",
                line={"width": 1.4},
                name=f"score ({res.strategy})",
                showlegend=False,
            ),
            row=i,
            col=1,
        )

        residual_mean = (mats.test_true - mats.test_pred).abs().mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=residual_mean.index,
                y=residual_mean.values,
                mode="lines",
                line={"width": 1, "dash": "dot"},
                name="mean |y_true-y_pred|",
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )

        if len(flagged_idx) > 0:
            flagged_scores = score_series.reindex(flagged_idx)
            fig.add_trace(
                go.Scatter(
                    x=flagged_scores.index,
                    y=flagged_scores.values,
                    mode="markers",
                    marker={"size": 7, "color": "red", "symbol": "x"},
                    name="flagged",
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )

        for start, end in known_intervals:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="orange",
                opacity=0.10,
                line_width=0,
                row=i,
                col=1,
            )

        fig.update_yaxes(title_text="score", row=i, col=1)

    fig.update_yaxes(title_text="z-value", row=1, col=1)
    fig.update_xaxes(title_text="time", row=len(selected) + 1, col=1)
    fig.update_layout(
        title=(
            f"Panel {panel_id} | KMeans strategy comparison on scorer test window "
            f"(rows={len(mats.test_true)})"
        ),
        template="plotly_white",
        height=max(350 * len(selected), 650),
        margin={"l": 50, "r": 20, "t": 70, "b": 40},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_html = output_dir / f"kmeans_strategy_compare_panel_{panel_id}_{ts}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return out_html


def run(args: argparse.Namespace) -> Path:
    logger = logging.get_logger("KMeansStrategyCompare")
    cfg = load_config(Path(args.config))

    # Force KMeans scorer comparison regardless of current config scorer.
    cfg = dict(cfg)
    cfg["detect"] = dict(cfg["detect"])
    cfg["detect"]["scorer_name"] = "KMeansScorer"

    if args.model_timestamp:
        cfg["detect"]["model_timestamp"] = args.model_timestamp

    if args.hist_window is not None:
        cfg["detect"]["hist_window"] = args.hist_window

    if args.high_quantile is not None:
        cfg["detect"]["high_quantile"] = args.high_quantile

    dm = DataManager(cfg, logger)
    detector = AnomalyDetector(cfg, logger)

    all_panel_data = dm.load_processed_data()
    panel_ids = [args.panel_id] if args.panel_id else cfg["panels"]["panel_ids"]

    k_values = _parse_csv_numbers(args.k_values, cast=int)
    window_values = _parse_csv_numbers(args.window_values, cast=int)
    trim_quantiles = _parse_csv_numbers(args.trim_quantiles, cast=float)

    scorer_base_params = dict(cfg["detect"].get("scorer_params", {}))
    scorer_base_params.pop("k", None)
    scorer_base_params.pop("n_clusters", None)
    scorer_base_params.pop("window", None)

    rows: list[dict] = []
    panel_strategy_results: dict[str, list[StrategyResult]] = {}
    panel_mats: dict[str, PanelMatrices] = {}
    ts = generate_timestamp()

    for panel_id in panel_ids:
        if panel_id not in all_panel_data:
            logger.warning(f"Skipping panel {panel_id}: no processed data found")
            continue

        mats = _get_panel_matrices(panel_id, all_panel_data[panel_id], cfg, detector)
        panel_mats[panel_id] = mats
        panel_strategy_results.setdefault(panel_id, [])

        row_error = (mats.train_true - mats.train_pred).abs().mean(axis=1)
        known_mask = _build_known_anomaly_mask(mats.test_true.index, cfg)

        for strat in _iter_strategies(k_values, window_values, trim_quantiles):
            trim_q = strat["trim_q"]
            if trim_q is None:
                keep_mask = pd.Series(True, index=row_error.index)
                cutoff = np.nan
            else:
                cutoff = row_error.quantile(trim_q)
                keep_mask = row_error <= cutoff

            train_true_f = mats.train_true.loc[keep_mask]
            train_pred_f = mats.train_pred.loc[keep_mask]
            dropped = int((~keep_mask).sum())

            if len(train_true_f) < 10:
                logger.warning(
                    f"Panel {panel_id} strategy {strat['name']}: too few scorer-train rows "
                    f"after trimming ({len(train_true_f)}). Skipping."
                )
                continue

            scorer_params = dict(scorer_base_params)
            scorer_params["n_clusters"] = strat["k"]
            scorer_params["window"] = strat["window"]

            fad = ForecastingAnomalyDetector(
                scorer_name="KMeansScorer",
                scorer_params=scorer_params,
                high_quantile=cfg["detect"]["high_quantile"],
                normalize_scores=cfg["detect"].get("normalize_scores", True),
                normalization_quantile=cfg["detect"].get("normalization_quantile", 0.99),
            )

            scores_df, flags_df = fad.fit_score_detect(
                y_true_train=train_true_f,
                y_pred_train=train_pred_f,
                y_true_test=mats.test_true,
                y_pred_test=mats.test_pred,
            )

            n_flags = int(flags_df["anomaly_flag"].sum())
            flag_rate = float(flags_df["anomaly_flag"].mean())
            flagged = flags_df["anomaly_flag"].reindex(mats.test_true.index).fillna(0).astype(int)
            flags_in_known = int(((flagged > 0) & known_mask).sum())
            flags_outside_known = int(((flagged > 0) & (~known_mask)).sum())
            known_points = int(known_mask.sum())
            flag_precision_proxy = (
                float(flags_in_known / n_flags) if n_flags > 0 else np.nan
            )

            rows.append(
                {
                    "panel_id": panel_id,
                    "strategy": strat["name"],
                    "k": strat["k"],
                    "window": strat["window"],
                    "trim_quantile": trim_q,
                    "trim_cutoff_mean_abs_error": cutoff,
                    "scorer_train_rows": int(len(train_true_f)),
                    "scorer_train_rows_dropped": dropped,
                    "scorer_test_rows": int(len(mats.test_true)),
                    "anomaly_flags": n_flags,
                    "anomaly_rate": flag_rate,
                    "flags_in_known_windows": flags_in_known,
                    "flags_outside_known_windows": flags_outside_known,
                    "known_window_points": known_points,
                    "flag_precision_proxy": flag_precision_proxy,
                    "score_mean": float(scores_df["anomaly_score"].mean()),
                    "score_q95": float(scores_df["anomaly_score"].quantile(0.95)),
                    "score_q99": float(scores_df["anomaly_score"].quantile(0.99)),
                    "score_norm_mean": float(scores_df["anomaly_score_normalized"].mean()),
                }
            )
            panel_strategy_results[panel_id].append(
                StrategyResult(
                    strategy=strat["name"],
                    k=strat["k"],
                    window=strat["window"],
                    trim_quantile=trim_q,
                    anomaly_flags=n_flags,
                    anomaly_rate=flag_rate,
                    scores_df=scores_df.copy(),
                    flags_df=flags_df.copy(),
                )
            )

    if not rows:
        raise RuntimeError("No strategy results produced.")

    out_df = pd.DataFrame(rows).sort_values(
        ["panel_id", "flags_outside_known_windows", "anomaly_flags", "anomaly_rate", "score_q99"],
        ascending=[True, True, True, True, True],
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(cfg["paths"]["results_dir"]) / f"kmeans_strategy_compare_{ts}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    # concise terminal preview
    preview_cols = [
        "panel_id",
        "strategy",
        "scorer_train_rows",
        "scorer_train_rows_dropped",
        "anomaly_flags",
        "anomaly_rate",
        "flags_in_known_windows",
        "flags_outside_known_windows",
        "flag_precision_proxy",
    ]
    print("\nTop strategies (fewest flags first):")
    print(out_df[preview_cols].head(20).to_string(index=False))
    print(f"\nSaved full comparison to: {output_path}")

    if args.plot:
        plot_dir = Path(args.plot_output_dir) if args.plot_output_dir else output_path.parent
        for panel_id in panel_ids:
            if panel_id not in panel_strategy_results or panel_id not in panel_mats:
                continue
            html_path = _save_panel_plot(
                panel_id=panel_id,
                mats=panel_mats[panel_id],
                strategy_results=panel_strategy_results[panel_id],
                cfg=cfg,
                output_dir=plot_dir,
                ts=ts,
                top_n=args.plot_top_n,
                context_columns=(
                    [c.strip() for c in args.context_columns.split(",") if c.strip()]
                    if args.context_columns
                    else None
                ),
                context_max_cols=args.context_max_cols,
            )
            print(f"Saved strategy plot for panel {panel_id}: {html_path}")
            detail_path = _save_panel_detail_csv(
                panel_id=panel_id,
                mats=panel_mats[panel_id],
                strategy_results=panel_strategy_results[panel_id],
                output_dir=plot_dir,
                ts=ts,
            )
            print(f"Saved per-timestamp detail for panel {panel_id}: {detail_path}")

    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare KMeans scorer strategies with optional trimming")
    p.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    p.add_argument("--panel-id", default=None, help="Optional single panel id (e.g., 1)")
    p.add_argument("--model-timestamp", default=None, help="Optional model timestamp to evaluate")
    p.add_argument("--hist-window", type=int, default=None, help="Override detect.hist_window")
    p.add_argument(
        "--high-quantile",
        type=float,
        default=None,
        help="Override detect.high_quantile used to flag anomalies (e.g. 0.999)",
    )
    p.add_argument("--k-values", default="2,3,4,5", help="Comma-separated n_clusters values")
    p.add_argument("--window-values", default="1", help="Comma-separated window values")
    p.add_argument(
        "--trim-quantiles",
        default="0.99,0.995,0.999",
        help="Comma-separated upper quantiles of scorer-train mean absolute error to keep",
    )
    p.add_argument("--output", default=None, help="Output CSV path")
    p.add_argument("--plot", action="store_true", help="Also create interactive HTML comparison plots")
    p.add_argument(
        "--plot-top-n",
        type=int,
        default=6,
        help="Number of best (fewest flags) strategies to visualize per panel",
    )
    p.add_argument(
        "--plot-output-dir",
        default=None,
        help="Directory for HTML plots (defaults to CSV output directory)",
    )
    p.add_argument(
        "--context-columns",
        default="",
        help="Comma-separated test columns to show in context plot (default: first N columns)",
    )
    p.add_argument(
        "--context-max-cols",
        type=int,
        default=6,
        help="Maximum number of panel columns shown in context plot when --context-columns is empty",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
