"""Tune per-channel forecaster configs for anomaly detection.

Selects the best (model, lags) per channel via one-step-ahead holdout MAE
on blocked folds.  Training-data contamination is handled at train time by
the adapter's ``auto_clean_anomalies`` option — this script only needs to
find the model + lags that track normal behaviour best.

Output:
    config/channel_models/panel_{id}.yaml

Usage:
    uv run python notebooks/tune_model.py --panel-id 1
    uv run python notebooks/tune_model.py --panel-id 1 --candidate-models LightGBM,Ridge
    uv run python notebooks/tune_model.py  # tunes all panels
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spotanomaly2.application.config import load_config
from spotanomaly2.application.data_manager import DataManager
from spotanomaly2.infrastructure.logging import get_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANDIDATE_MODELS: dict[str, dict[str, Any]] = {
    "Ridge": {"alpha": 1.0},
    "LightGBM": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
    },
}

LAG_CANDIDATES = [6, 12, 24, 48, 168]


def _build_estimator(name: str, params: dict[str, Any], seed: int) -> Any:
    n = name.strip().lower()
    if n == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    if n in ("lightgbm", "lgbm"):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, verbose=-1, n_jobs=1, **params)
    if n in ("xgboost", "xgb"):
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, verbosity=0, n_jobs=1, **params)
    if n == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=seed, verbose=0, thread_count=1, **params)
    raise ValueError(f"Unknown model: {name!r}")


def _select_targets(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if not c.endswith("__weight")
        and not c.startswith("weather_")
        and not c.startswith("exogenous_")
    ]


def _ensure_freq(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.freq is not None:
        return df
    inferred = pd.infer_freq(df.index)
    if inferred is None:
        deltas = np.diff(df.index.values).astype("timedelta64[s]").astype(int)
        if deltas.size == 0:
            return df
        inferred = pd.tseries.frequencies.to_offset(pd.Timedelta(seconds=int(np.median(deltas))))
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index, freq=inferred)
    return df


def _blocked_folds(n: int, n_folds: int, val_size: int) -> list[tuple[slice, slice]]:
    """Non-overlapping (train, val) slices from the tail of the series."""
    folds: list[tuple[slice, slice]] = []
    end = n
    for _ in range(n_folds):
        val_start = end - val_size
        if val_start <= val_size:
            break
        folds.append((slice(0, val_start), slice(val_start, end)))
        end = val_start
    return list(reversed(folds))


def _onestep_mae(
    estimator: Any,
    lags: list[int],
    y_train: pd.Series,
    y_val: pd.Series,
) -> float:
    """Fit on y_train, one-step-ahead MAE on y_val using real observed lags."""
    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    forecaster = ForecasterRecursive(estimator=estimator, lags=lags)
    forecaster.fit(y=y_train)

    y_all = pd.concat([y_train, y_val])
    X_feat, y_aligned = forecaster.create_train_X_y(y=y_all)
    val_mask = y_aligned.index.isin(y_val.index)
    if val_mask.sum() < 4:
        return float("inf")

    inner = getattr(forecaster, "regressor", None) or forecaster.estimator
    preds = np.asarray(inner.predict(X_feat.loc[val_mask]), dtype=float)
    actual = np.asarray(y_aligned.loc[val_mask].values, dtype=float)
    ok = ~(np.isnan(preds) | np.isnan(actual))
    if ok.sum() < 4:
        return float("inf")
    return float(np.mean(np.abs(actual[ok] - preds[ok])))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _data_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(str(df.index.min()).encode())
    h.update(str(df.index.max()).encode())
    h.update(",".join(sorted(df.columns)).encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Per-channel tuning
# ---------------------------------------------------------------------------

def tune_channel(
    target: str,
    y: pd.Series,
    candidate_models: dict[str, dict[str, Any]],
    lag_candidates: list[int],
    folds: list[tuple[slice, slice]],
    seed: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Grid search over (model, lags) scored by mean one-step-ahead MAE."""
    best_score = float("inf")
    best_model = "Ridge"
    best_params: dict[str, Any] = {"alpha": 1.0}
    best_lags: list[int] = list(range(1, 7))

    for model_name, params in candidate_models.items():
        for max_lag in lag_candidates:
            if max_lag >= len(y) // 3:
                continue
            lags = list(range(1, max_lag + 1))
            fold_scores: list[float] = []
            for train_sl, val_sl in folds:
                try:
                    est = _build_estimator(model_name, params, seed)
                    mae = _onestep_mae(est, lags, y.iloc[train_sl], y.iloc[val_sl])
                    if np.isfinite(mae):
                        fold_scores.append(mae)
                except (ValueError, RuntimeError, ImportError) as exc:
                    logger.warning("  %s lag=%d fold failed: %s", model_name, max_lag, exc)

            if not fold_scores:
                continue
            mean_mae = float(np.mean(fold_scores))
            logger.info("  %s  lag=%3d  MAE=%.4f (±%.4f, %d folds)",
                        model_name.ljust(10), max_lag, mean_mae,
                        float(np.std(fold_scores)), len(fold_scores))
            if mean_mae < best_score:
                best_score = mean_mae
                best_model = model_name
                best_params = dict(params)
                best_lags = lags

    return {
        "model": best_model,
        "params": best_params,
        "best_lags": best_lags,
        "mae": round(best_score, 6),
    }


# ---------------------------------------------------------------------------
# Panel orchestration
# ---------------------------------------------------------------------------

def tune_panel(
    panel_id: str,
    panel_df: pd.DataFrame,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Path:
    panel_df = _ensure_freq(panel_df)
    panel_df = panel_df.copy()
    panel_df.index.name = "DateTime"
    targets = _select_targets(panel_df)

    n = len(panel_df)
    folds = _blocked_folds(n, n_folds=args.n_folds, val_size=args.val_size)
    if len(folds) < 2:
        raise ValueError(f"Panel {panel_id}: not enough data for {args.n_folds} folds "
                         f"of {args.val_size} points")
    logger.info("Panel %s: %d targets, %d folds, val_size=%d",
                panel_id, len(targets), len(folds), args.val_size)

    models = {}
    for name in args.candidate_models:
        if name in CANDIDATE_MODELS:
            models[name] = CANDIDATE_MODELS[name]
        else:
            models[name] = {}

    channels: dict[str, Any] = {}
    for target in targets:
        y = panel_df[target].astype(float)
        if y.std() < 1e-9:
            logger.warning("  %s: near-constant, skipping", target)
            continue
        logger.info("Tuning %s ...", target)
        result = tune_channel(
            target=target,
            y=y,
            candidate_models=models,
            lag_candidates=args.lag_candidates,
            folds=folds,
            seed=cfg.get("train", {}).get("random_seed", 42),
            logger=logger,
        )
        logger.info("  -> %s  lags=%d  MAE=%.4f",
                     result["model"], max(result["best_lags"]), result["mae"])
        channels[target] = {
            "model": result["model"],
            "params": result["params"],
            "best_lags": result["best_lags"],
        }

    payload = {
        "provenance": {
            "git_sha": _git_sha(),
            "data_fingerprint": _data_fingerprint(panel_df),
            "n_folds": args.n_folds,
            "val_size": args.val_size,
            "candidate_models": args.candidate_models,
            "lag_candidates": args.lag_candidates,
        },
        "channels": channels,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"panel_{panel_id}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    logger.info("Wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune per-channel forecaster configs")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--panel-id", default=None,
                   help="Panel ID to tune. Omit to tune all panels.")
    p.add_argument("--output-dir", default="config/channel_models")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--val-size", type=int, default=168,
                   help="Validation size per fold in data points (default 168).")
    p.add_argument("--candidate-models", type=_parse_list,
                   default=["Ridge", "LightGBM"],
                   help="Comma-separated model names (default: Ridge,LightGBM).")
    p.add_argument("--lag-candidates", type=_parse_ints,
                   default=[6, 12, 24, 48, 168],
                   help="Comma-separated max-lag values to try.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(Path(args.config))
    logger = get_logger("TuneModels")
    logger.setLevel(logging.INFO)

    dm = DataManager(cfg, logger)
    panel_data = dm.load_processed_data()
    panel_ids = [args.panel_id] if args.panel_id else cfg.get("panels", {}).get("panel_ids", [])
    if not panel_ids:
        raise RuntimeError("No panel IDs found")

    for panel_id in panel_ids:
        if panel_id not in panel_data:
            logger.warning("Skipping panel %s: no processed data", panel_id)
            continue
        tune_panel(panel_id, panel_data[panel_id], cfg, args, logger)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
