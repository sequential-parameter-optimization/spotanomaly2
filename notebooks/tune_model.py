"""Tune panel forecasting models using the SpotForecast2 Optuna task.

This script uses the SpotForecast2 tuning task (OptunaTask) to tune the
recursive forecasters for each panel in the SpotAnomaly workspace.

What it does:
- Loads processed panel data from data/processed
- Selects only real sensor columns as forecast targets
- Runs SpotForecast2 Optuna tuning per panel
- Persists tuned models via SpotForecast2's cache
- Writes a panel-specific YAML config with tuned parameters
- Writes a CSV summary for quick inspection

Typical usage:
    uv run python tune_model.py \
    --config config/config.yaml \
      --panel-id 1 \
      --n-trials-optuna 10 \
      --train-days 180 \
      --val-days 14 \
      --predict-size 24 \
      --show false

Output:
- config/channel_models/panel_1_optuna.yaml
- data/results/tuning/panel_1_optuna_summary.csv
- SpotForecast2 cache directory with tuned model artifacts
"""

from __future__ import annotations

import argparse
import csv
import ast
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from spotanomaly2.application.config import load_config
from spotanomaly2.application.data_manager import DataManager
import logging
from spotanomaly2.infrastructure.logging import get_logger
from spotanomaly2.infrastructure.storage import generate_timestamp
from spotforecast2.manager.multitask import OptunaTask, SpotOptimTask


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _select_target_columns(df: pd.DataFrame) -> list[str]:
    """Select actual sensor targets from a processed panel DataFrame."""
    return [
        c
        for c in df.columns
        if not c.endswith("__weight")
        and not c.startswith("weather_")
        and not c.startswith("exogenous_")
    ]


def _sanitize_best_params(best_params: dict[str, Any]) -> dict[str, Any]:
    """Remove search-only keys that are not model hyperparameters."""
    params = dict(best_params)
    params.pop("lags", None)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Model-selection helpers – evaluate multiple estimator types per channel
# ─────────────────────────────────────────────────────────────────────────────

def _build_candidate_estimator(model_name: str, random_state: int = 42) -> Any:
    """Instantiate a scikit-learn compatible estimator by name."""
    name = model_name.strip().lower()
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=random_state, verbose=-1)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=random_state, verbosity=0, n_jobs=1)
    if name in ("ridge", "linear", "linearregression"):
        from sklearn.linear_model import Ridge
        return Ridge()
    if name == "catboost":
        from catboost import CatBoostRegressor

        class _CatBoostRegressor(CatBoostRegressor):
            @property
            def task_type(self):
                return "CPU"

            def set_params(self, **params):
                try:
                    return super().set_params(**params)
                except Exception:
                    return self

        return _CatBoostRegressor(random_seed=random_state, verbose=0)
    raise ValueError(f"Unknown candidate model: {model_name!r}")


def _make_rolling_window_features(window_size: list[int] | None) -> Any:
    """Try to create a RollingFeaturesUnified instance; return None if unavailable."""
    if not window_size:
        return None
    for module_path in (
        "spotforecast2_safe.feature_engineering.rolling",
        "spotforecast2_safe.feature_engineering",
    ):
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, "RollingFeaturesUnified", None)
            if cls is not None:
                return cls(stats=["mean"], window_sizes=window_size)
        except (ImportError, AttributeError):
            pass
    return None


def _make_comparison_forecaster(model_name: str, task: Any, max_lags: int) -> Any:
    """Build a ForecasterRecursive for model comparison using the task's config."""
    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    task_cfg = getattr(task, "config", None)
    random_state = int(getattr(task_cfg, "random_state", 42))
    window_size = getattr(task_cfg, "window_size", None)
    weight_func = getattr(task, "weight_func", None)

    return ForecasterRecursive(
        estimator=_build_candidate_estimator(model_name, random_state),
        lags=max_lags,
        window_features=_make_rolling_window_features(window_size),
        weight_func=weight_func,
    )


def _comparison_search_space(
    model_name: str,
    lag_candidates: list[int],
    lgbm_best_params: dict[str, Any] | None = None,
) -> Any:
    """Return an Optuna search-space callable for the given model type."""
    name = model_name.strip().lower()

    if name == "lightgbm":
        bp = lgbm_best_params or {}
        n_est = int(bp.get("n_estimators", 200))
        lr = float(bp.get("learning_rate", 0.05))
        leaves = int(bp.get("num_leaves", 31))
        n_lo, n_hi = max(50, n_est - 100), min(2000, n_est + 100)
        lr_lo, lr_hi = max(1e-4, lr * 0.5), min(0.3, lr * 2.0)
        l_lo, l_hi = max(8, leaves - 16), min(256, leaves + 16)

        def space_lgbm(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", n_lo, n_hi),
                "learning_rate": trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
                "num_leaves": trial.suggest_int("num_leaves", l_lo, l_hi),
                "lags": trial.suggest_categorical("lags", lag_candidates),
            }
        return space_lgbm

    if name == "xgboost":
        def space_xgb(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "lags": trial.suggest_categorical("lags", lag_candidates),
            }
        return space_xgb

    if name in ("ridge", "linear", "linearregression"):
        def space_ridge(trial):
            return {
                "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
                "lags": trial.suggest_categorical("lags", lag_candidates),
            }
        return space_ridge

    if name == "catboost":
        def space_catboost(trial):
            return {
                "iterations": trial.suggest_int("iterations", 50, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
                "lags": trial.suggest_categorical("lags", lag_candidates),
            }
        return space_catboost

    raise ValueError(f"No comparison search space defined for model: {model_name!r}")


def _evaluate_model_candidate(
    task: Any,
    target: str,
    model_name: str,
    lag_candidates: list[int],
    n_trials: int,
    metric: str,
    lgbm_best_params: dict[str, Any] | None,
    logger: Any,
) -> dict[str, Any] | None:
    """Run Bayesian search for one model × target; return a result dict or None."""
    try:
        from spotforecast2.model_selection import bayesian_search_forecaster

        y_train, exog_train, _ = task._get_target_data(target)
        cv = task.cv_ts(y_train)

        exog = (
            None
            if exog_train is None or (hasattr(exog_train, "empty") and exog_train.empty)
            else exog_train
        )
        max_lags = max(lag_candidates) if lag_candidates else 24
        forecaster = _make_comparison_forecaster(model_name, task, max_lags)
        search_space = _comparison_search_space(model_name, lag_candidates, lgbm_best_params)

        results_df, _ = bayesian_search_forecaster(
            forecaster=forecaster,
            y=y_train,
            cv=cv,
            search_space=search_space,
            metric=metric,
            exog=exog,
            n_trials=n_trials,
            return_best=False,
            verbose=False,
            show_progress=False,
            suppress_warnings=True,
        )

        if results_df is None or results_df.empty:
            return None

        # Locate the metric column (name may vary slightly)
        if metric in results_df.columns:
            metric_col = metric
        else:
            metric_col = next(
                (c for c in results_df.columns if "error" in c.lower() or "loss" in c.lower()),
                None,
            )
            if metric_col is None:
                return None

        best_row = results_df.loc[results_df[metric_col].idxmin()]
        best_score = float(best_row[metric_col])

        best_lags = None
        if "lags" in best_row.index:
            raw = best_row["lags"]
            best_lags = raw.tolist() if hasattr(raw, "tolist") else raw

        # Prefer the 'params' dict column; fall back to individual columns
        params_val = best_row.get("params") if "params" in best_row.index else None
        if isinstance(params_val, dict):
            best_params: dict[str, Any] = {k: v for k, v in params_val.items() if k != "lags"}
        else:
            skip = {"lags", "params", "lags_label", metric_col}
            best_params = {
                col: best_row[col]
                for col in results_df.columns
                if col not in skip
                and not isinstance(best_row[col], (pd.Series, pd.DataFrame))
            }

        return {
            "target": target,
            "model": model_name,
            "best_score": best_score,
            "best_lags": best_lags,
            "best_params": best_params,
        }

    except Exception as exc:
        logger.warning("Model comparison: %s/%s failed: %s", model_name, target, exc)
        return None


def _run_model_selection(
    task: Any,
    tuned_lgbm_results: list[dict[str, Any]],
    candidate_models: list[str],
    n_trials: int,
    metric: str,
    lag_candidates: list[int],
    logger: Any,
) -> list[dict[str, Any]]:
    """Evaluate all candidate models per target; return updated results with winner chosen.

    Each returned entry gains a ``model`` key (winning model name) and a
    ``comparison_scores`` dict mapping every evaluated model to its best score.
    """
    final: list[dict[str, Any]] = []

    for lgbm_result in tuned_lgbm_results:
        target = lgbm_result["target"]
        lgbm_params = _sanitize_best_params(lgbm_result.get("best_params", {}))
        model_scores: dict[str, dict[str, Any]] = {}

        for model_name in candidate_models:
            logger.info(
                "  model-selection: target=%s  model=%s  trials=%d  metric=%s",
                target, model_name, n_trials, metric,
            )
            result = _evaluate_model_candidate(
                task=task,
                target=target,
                model_name=model_name,
                lag_candidates=lag_candidates,
                n_trials=n_trials,
                metric=metric,
                lgbm_best_params=lgbm_params if model_name.lower() == "lightgbm" else None,
                logger=logger,
            )
            if result is not None:
                model_scores[model_name] = result

        if not model_scores:
            logger.warning(
                "All comparisons failed for target %s; keeping LightGBM result.", target
            )
            final.append({**lgbm_result, "model": "LightGBM", "comparison_scores": {}})
            continue

        winner = min(model_scores, key=lambda m: model_scores[m]["best_score"])
        best = model_scores[winner]
        logger.info(
            "  model-selection: target=%s  winner=%s  score=%.6f  all=%s",
            target, winner, best["best_score"],
            {m: f"{v['best_score']:.6f}" for m, v in model_scores.items()},
        )
        final.append({
            "target": target,
            "model": winner,
            "best_score": best["best_score"],
            "best_lags": best.get("best_lags", lgbm_result.get("best_lags")),
            "best_params": best.get("best_params", lgbm_params),
            "task_name": lgbm_result.get("task_name", "optuna"),
            "timestamp": lgbm_result.get("timestamp"),
            "comparison_scores": {m: v["best_score"] for m, v in model_scores.items()},
        })

    return final


# ─────────────────────────────────────────────────────────────────────────────

def _load_panel_baseline_config(cfg: dict[str, Any], panel_id: str) -> dict[str, Any]:
    """Load the panel-specific channel model config used as tuning baseline."""
    train_cfg = cfg.get("train", {})
    file_map = train_cfg.get("channel_config_files", {})
    cfg_path_value = file_map.get(panel_id) or file_map.get(f"panel_{panel_id}")
    if not cfg_path_value:
        return {}

    cfg_path = _resolve_path(str(cfg_path_value), Path.cwd())
    if not cfg_path.exists():
        raise FileNotFoundError(f"Panel config not found for panel {panel_id}: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        panel_cfg = yaml.safe_load(f) or {}
    if not isinstance(panel_cfg, dict):
        raise ValueError(f"Panel config must be a mapping: {cfg_path}")
    return panel_cfg


def _baseline_values(cfg: dict[str, Any], panel_id: str) -> tuple[dict[str, Any], int]:
    """Return baseline hyperparameters and lag count for a panel."""
    panel_cfg = _load_panel_baseline_config(cfg, panel_id)
    default_section = panel_cfg.get("default", {}) if isinstance(panel_cfg, dict) else {}
    if not isinstance(default_section, dict):
        default_section = {}

    baseline_params = dict(default_section.get("params", {}) or {})
    baseline_lags = int(cfg.get("train", {}).get("lags", 6))
    recommended_lags = panel_cfg.get("recommended_lags")
    if recommended_lags is not None:
        if isinstance(recommended_lags, int):
            baseline_lags = recommended_lags
        elif isinstance(recommended_lags, (list, tuple)) and recommended_lags:
            baseline_lags = int(recommended_lags[0])
    return baseline_params, baseline_lags


def _resolve_channel_baseline(
    panel_cfg: dict[str, Any],
    target: str,
    fallback_model: str,
) -> tuple[str, dict[str, Any]]:
    """Return baseline (model, params) for a channel target from panel config."""
    default_section = panel_cfg.get("default", {}) if isinstance(panel_cfg, dict) else {}
    channels_section = panel_cfg.get("channels", {}) if isinstance(panel_cfg, dict) else {}
    if not isinstance(default_section, dict):
        default_section = {}
    if not isinstance(channels_section, dict):
        channels_section = {}

    default_model = str(default_section.get("model", fallback_model))
    default_params = dict(default_section.get("params", {}) or {})

    channel_entry = channels_section.get(target, {})
    if not isinstance(channel_entry, dict):
        channel_entry = {}

    model_name = str(channel_entry.get("model", default_model))
    params = dict(channel_entry.get("params", default_params) or {})
    return model_name, params


def _lightgbm_targets(
    target_columns: list[str],
    panel_cfg: dict[str, Any],
    fallback_model: str,
) -> list[str]:
    """Return only channels configured to use LightGBM."""
    selected: list[str] = []
    for target in target_columns:
        model_name, _ = _resolve_channel_baseline(panel_cfg, target, fallback_model)
        if model_name.strip().lower() == "lightgbm":
            selected.append(target)
    return selected


def _build_optuna_search_space(
    baseline_params: dict[str, Any],
    baseline_lags: int,
) -> Any:
    """Build a narrow Optuna search space around the provided baseline."""

    def _int_bounds(center: int, min_value: int, max_value: int, lo: float = 0.5, hi: float = 2.0) -> tuple[int, int]:
        low = max(min_value, int(round(center * lo)))
        high = min(max_value, int(round(center * hi)))
        if low > high:
            low = high
        return low, high

    def _float_bounds(center: float, min_value: float, max_value: float, lo: float = 0.5, hi: float = 2.0) -> tuple[float, float]:
        low = max(min_value, center * lo)
        high = min(max_value, center * hi)
        if low >= high:
            low = min_value
            high = max_value
        return low, high

    n_estimators_base = int(baseline_params.get("n_estimators", 300))
    learning_rate_base = float(baseline_params.get("learning_rate", 0.05))
    num_leaves_base = int(baseline_params.get("num_leaves", 31))

    n_estimators_low, n_estimators_high = _int_bounds(n_estimators_base, 50, 2000)
    num_leaves_low, num_leaves_high = _int_bounds(num_leaves_base, 8, 256)
    lr_low, lr_high = _float_bounds(learning_rate_base, 1e-4, 0.3, lo=0.2, hi=5.0)

    lag_candidates = sorted({
        max(1, baseline_lags // 2),
        max(1, baseline_lags - 1),
        baseline_lags,
        baseline_lags + 1,
        max(1, baseline_lags * 2),
    })

    def search_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", n_estimators_low, n_estimators_high),
            "learning_rate": trial.suggest_float("learning_rate", lr_low, lr_high, log=True),
            "num_leaves": trial.suggest_int("num_leaves", num_leaves_low, num_leaves_high),
            "lags": trial.suggest_categorical("lags", lag_candidates),
        }

    return search_space


def _build_spotoptim_search_space(
    baseline_params: dict[str, Any],
    baseline_lags: int,
) -> dict[str, Any]:
    """Build a narrow SpotOptim search space around the provided baseline."""

    def _int_bounds(center: int, min_value: int, max_value: int, lo: float = 0.5, hi: float = 2.0) -> tuple[int, int]:
        low = max(min_value, int(round(center * lo)))
        high = min(max_value, int(round(center * hi)))
        if low > high:
            low = high
        return low, high

    def _float_bounds(center: float, min_value: float, max_value: float, lo: float = 0.5, hi: float = 2.0) -> tuple[float, float, str]:
        low = max(min_value, center * lo)
        high = min(max_value, center * hi)
        if low >= high:
            low = min_value
            high = max_value
        return low, high, "log10"

    n_estimators_base = int(baseline_params.get("n_estimators", 300))
    learning_rate_base = float(baseline_params.get("learning_rate", 0.05))
    num_leaves_base = int(baseline_params.get("num_leaves", 31))

    n_estimators_low, n_estimators_high = _int_bounds(n_estimators_base, 50, 2000)
    num_leaves_low, num_leaves_high = _int_bounds(num_leaves_base, 8, 256)
    lr_low, lr_high, lr_scale = _float_bounds(learning_rate_base, 1e-4, 0.3, lo=0.2, hi=5.0)

    lag_candidates = [
        str(v) for v in sorted({
            max(1, baseline_lags // 2),
            max(1, baseline_lags - 1),
            baseline_lags,
            baseline_lags + 1,
            max(1, baseline_lags * 2),
        })
    ]

    return {
        "n_estimators": (n_estimators_low, n_estimators_high),
        "learning_rate": (lr_low, lr_high, lr_scale),
        "num_leaves": (num_leaves_low, num_leaves_high),
        "lags": lag_candidates,
    }


def _pick_default_target(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the default target summary for the panel YAML.

    We prefer the target with the lowest validation score if present, otherwise
    the first successful result.
    """
    scored = [r for r in results if r.get("best_score") is not None]
    if scored:
        return sorted(scored, key=lambda r: r["best_score"])[0]
    return results[0]


def _build_optuna_task(
    panel_id: str,
    panel_df: pd.DataFrame,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    target_columns: list[str],
    cache_home: Path,
) -> OptunaTask:
    weather_cfg = cfg.get("process", {}).get("weather", {})
    return OptunaTask(
        dataframe=panel_df,
        data_frame_name=f"panel_{panel_id}",
        cache_home=cache_home,
        predict_size=args.predict_size,
        n_trials_optuna=args.n_trials_optuna,
        train_days=args.train_days,
        val_days=args.val_days,
        number_folds=args.number_folds,
        contamination=args.contamination,
        imputation_method=args.imputation_method,
        use_exogenous_features=args.use_exogenous_features,
        auto_save_models=True,
        log_level=logging.INFO if args.verbose else logging.WARNING,
        verbose=args.verbose,
        country_code=args.country_code,
        state=args.state,
        timezone=args.timezone,
        latitude=weather_cfg.get("latitude", args.latitude),
        longitude=weather_cfg.get("longitude", args.longitude),
        targets=target_columns,
    )


def _build_spotoptim_task(
    panel_id: str,
    panel_df: pd.DataFrame,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    target_columns: list[str],
    cache_home: Path,
) -> SpotOptimTask:
    weather_cfg = cfg.get("process", {}).get("weather", {})
    return SpotOptimTask(
        dataframe=panel_df,
        data_frame_name=f"panel_{panel_id}",
        cache_home=cache_home,
        predict_size=args.predict_size,
        n_trials_spotoptim=args.n_trials_spotoptim,
        n_initial_spotoptim=args.n_initial_spotoptim,
        train_days=args.train_days,
        val_days=args.val_days,
        number_folds=args.number_folds,
        contamination=args.contamination,
        imputation_method=args.imputation_method,
        use_exogenous_features=args.use_exogenous_features,
        auto_save_models=True,
        log_level=logging.INFO if args.verbose else logging.WARNING,
        verbose=args.verbose,
        country_code=args.country_code,
        state=args.state,
        timezone=args.timezone,
        latitude=weather_cfg.get("latitude", args.latitude),
        longitude=weather_cfg.get("longitude", args.longitude),
        targets=target_columns,
    )


def _load_tuning_result(task: OptunaTask, target: str, task_name: str) -> dict[str, Any] | None:
    result = task.load_tuning_results(target=target, task_name=task_name)
    if result is None:
        return None
    return {
        "target": target,
        "best_params": result.get("best_params", {}),
        "best_lags": result.get("best_lags"),
        "best_score": result.get("best_score"),
        "task_name": result.get("task_name", task_name),
        "timestamp": result.get("timestamp"),
    }


def _write_panel_yaml(
    panel_id: str,
    panel_cfg: dict[str, Any],
    all_targets: list[str],
    results: list[dict[str, Any]],
    fallback_model: str,
    fallback_lags: int,
    args: argparse.Namespace,
    output_dir: Path,
    task_name: str,
) -> Path:
    default_section = panel_cfg.get("default", {}) if isinstance(panel_cfg, dict) else {}
    baseline_default_model = str((default_section or {}).get("model", fallback_model))
    baseline_default_params = dict((default_section or {}).get("params", {}) or {})

    default_result = _pick_default_target(results) if results else None
    default_params = (
        _sanitize_best_params(default_result["best_params"])
        if default_result is not None
        else baseline_default_params
    )
    default_model = (
        default_result.get("model", "LightGBM") if default_result is not None else baseline_default_model
    )
    default_lags = default_result.get("best_lags") if default_result is not None else None
    tuned_by_target = {item["target"]: item for item in results}

    channel_entries: dict[str, Any] = {}
    lag_counter: Counter[str] = Counter()
    for target in all_targets:
        base_model, base_params = _resolve_channel_baseline(panel_cfg, target, fallback_model)
        tuned = tuned_by_target.get(target)
        if tuned is not None and base_model.strip().lower() == "lightgbm":
            best_lags = tuned.get("best_lags")
            if best_lags is not None:
                lag_counter[repr(best_lags)] += 1
            winner_model = tuned.get("model", "LightGBM")
            channel_entries[target] = {
                "model": winner_model,
                "params": _sanitize_best_params(tuned["best_params"]),
                "best_lags": best_lags,
            }
        else:
            channel_entries[target] = {
                "model": base_model,
                "params": base_params,
            }

    panel_recommended_lags = panel_cfg.get("recommended_lags") if isinstance(panel_cfg, dict) else None
    recommended_lags = default_lags or panel_recommended_lags or fallback_lags
    if lag_counter:
        recommended_lags = ast.literal_eval(lag_counter.most_common(1)[0][0])

    source_map = {
        "optuna": "spotforecast2.manager.multitask.OptunaTask",
        "spotoptim": "spotforecast2.manager.multitask.SpotOptimTask",
    }
    n_trials = args.n_trials_optuna if task_name == "optuna" else args.n_trials_spotoptim
    payload = {
        "task": task_name,
        "source": source_map.get(task_name, task_name),
        "panel_id": panel_id,
        "predict_size": args.predict_size,
        "n_trials": n_trials,
        "train_days": args.train_days,
        "val_days": args.val_days,
        "recommended_lags": recommended_lags,
        "default": {
            "model": default_model,
            "params": default_params,
        },
        "channels": channel_entries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"panel_{panel_id}_{task_name}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return out_path


def _write_summary_csv(panel_id: str, results: list[dict[str, Any]], output_dir: Path, task_name: str) -> Path:
    rows = []
    for item in results:
        rows.append(
            {
                "panel_id": panel_id,
                "target": item["target"],
                "best_model": item.get("model", "LightGBM"),
                "best_score": item.get("best_score"),
                "best_lags": json.dumps(item.get("best_lags"), default=str),
                "best_params": json.dumps(_sanitize_best_params(item.get("best_params", {})), default=str),
                "comparison_scores": json.dumps(item.get("comparison_scores", {}), default=str),
                "task_name": item.get("task_name", "optuna"),
                "timestamp": item.get("timestamp"),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"panel_{panel_id}_{task_name}_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "panel_id", "target", "best_model", "best_score",
            "best_lags", "best_params", "comparison_scores", "task_name", "timestamp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return out_path


def tune_panel(
    panel_id: str,
    panel_df: pd.DataFrame,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    results_dir: Path,
    logger,
) -> tuple[Path, Path]:
    panel_df = panel_df.copy()
    if not isinstance(panel_df.index, pd.DatetimeIndex):
        raise ValueError(f"Panel {panel_id} must have a DatetimeIndex")

    panel_df.index.name = args.index_name
    target_columns = _select_target_columns(panel_df)
    if not target_columns:
        raise ValueError(f"Panel {panel_id}: no target columns found")

    panel_cfg = _load_panel_baseline_config(cfg, panel_id)
    fallback_model = str(cfg.get("train", {}).get("model", "LightGBM"))
    tune_targets = _lightgbm_targets(target_columns, panel_cfg, fallback_model)
    skipped_targets = [t for t in target_columns if t not in tune_targets]

    logger.info("Panel %s: all targets: %s", panel_id, ", ".join(target_columns))
    if tune_targets:
        logger.info("Panel %s: LightGBM targets to tune: %s", panel_id, ", ".join(tune_targets))
    if skipped_targets:
        logger.info(
            "Panel %s: preserving baseline model/params for non-LightGBM targets: %s",
            panel_id,
            ", ".join(skipped_targets),
        )

    cache_home = Path(cfg["paths"]["models_dir"]) / "spotforecast2_tuning_cache"
    task: OptunaTask | SpotOptimTask | None = None
    if args.task == "spotoptim":
        task_name = "spotoptim"
        if tune_targets:
            task = _build_spotoptim_task(
                panel_id=panel_id,
                panel_df=panel_df,
                cfg=cfg,
                args=args,
                target_columns=tune_targets,
                cache_home=cache_home,
            )
    else:
        task_name = "optuna"
        if tune_targets:
            task = _build_optuna_task(
                panel_id=panel_id,
                panel_df=panel_df,
                cfg=cfg,
                args=args,
                target_columns=tune_targets,
                cache_home=cache_home,
            )

    baseline_params, baseline_lags = _baseline_values(cfg, panel_id)
    if task_name == "spotoptim":
        search_space = _build_spotoptim_search_space(baseline_params, baseline_lags)
    else:
        search_space = _build_optuna_search_space(baseline_params, baseline_lags)

    if task is not None:
        # Make the target selection explicit before the task prepares the pipeline.
        task.config.targets = tune_targets

        logger.info(
            "Panel %s: starting %s tuning with %d trials per target (baseline lags=%s, params=%s)",
            panel_id,
            task_name,
            args.n_trials_optuna if task_name == "optuna" else args.n_trials_spotoptim,
            baseline_lags,
            baseline_params,
        )
        task.prepare_data()
        task.run(show=args.show, search_space=search_space)
    else:
        logger.info("Panel %s: no LightGBM targets found; writing baseline models unchanged", panel_id)

    tuned_results: list[dict[str, Any]] = []
    for target in tune_targets:
        if task is None:
            continue
        result = _load_tuning_result(task, target, task_name)
        if result is None:
            logger.warning("Panel %s: no tuning result found for %s", panel_id, target)
            continue
        tuned_results.append(result)

    # ── model selection ───────────────────────────────────────────────────────
    candidate_models = [
        m.strip()
        for m in getattr(args, "candidate_models", "LightGBM").split(",")
        if m.strip()
    ]
    n_trials_compare = getattr(args, "n_trials_compare", 5)
    compare_metric = getattr(args, "compare_metric", "mean_absolute_error")

    if task is not None and len(candidate_models) > 1 and tuned_results:
        lag_candidates_compare = sorted({
            max(1, baseline_lags // 2),
            max(1, baseline_lags - 1),
            baseline_lags,
            baseline_lags + 1,
            max(1, baseline_lags * 2),
        })
        logger.info(
            "Panel %s: model selection — candidates=%s, trials_each=%d, metric=%s",
            panel_id, candidate_models, n_trials_compare, compare_metric,
        )
        try:
            tuned_results = _run_model_selection(
                task=task,
                tuned_lgbm_results=tuned_results,
                candidate_models=candidate_models,
                n_trials=n_trials_compare,
                metric=compare_metric,
                lag_candidates=lag_candidates_compare,
                logger=logger,
            )
        except Exception as exc:
            logger.warning(
                "Panel %s: model selection failed (%s); keeping LightGBM results.", panel_id, exc,
            )
    # ─────────────────────────────────────────────────────────────────────────

    yaml_path = _write_panel_yaml(
        panel_id=panel_id,
        panel_cfg=panel_cfg,
        all_targets=target_columns,
        results=tuned_results,
        fallback_model=fallback_model,
        fallback_lags=baseline_lags,
        args=args,
        output_dir=output_dir,
        task_name=task_name,
    )
    csv_path = _write_summary_csv(panel_id, tuned_results, results_dir, task_name)

    logger.info("Panel %s: wrote tuned panel config to %s", panel_id, yaml_path)
    logger.info("Panel %s: wrote tuning summary to %s", panel_id, csv_path)
    return yaml_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune panel models using SpotForecast2 OptunaTask"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="spotoptim",
        choices=["optuna", "spotoptim"],
        help="Which SpotForecast2 tuning task to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the SpotAnomaly config file",
    )
    parser.add_argument(
        "--panel-id",
        type=str,
        default=None,
        help="Optional single panel id (e.g. 1). If omitted, all panels are tuned.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="config/channel_models",
        help="Directory for generated tuned panel YAML files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results/tuning",
        help="Directory for tuning summary CSV files",
    )
    parser.add_argument(
        "--n-trials-optuna",
        type=int,
        default=15,
        help="Number of Optuna trials per target",
    )
    parser.add_argument(
        "--n-trials-spotoptim",
        type=int,
        default=10,
        help="Number of SpotOptim trials per target",
    )
    parser.add_argument(
        "--n-initial-spotoptim",
        type=int,
        default=5,
        help="Number of initial SpotOptim evaluations",
    )
    parser.add_argument(
        "--predict-size",
        type=int,
        default=24,
        help="Forecast horizon used by SpotForecast2",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=180,
        help="Training window in days for the tuning task",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=14,
        help="Validation window in days for the tuning task",
    )
    parser.add_argument(
        "--number-folds",
        type=int,
        default=10,
        help="Number of folds used in the tuning task",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.03,
        help="Outlier contamination passed to SpotForecast2",
    )
    parser.add_argument(
        "--imputation-method",
        type=str,
        default="weighted",
        help="Imputation method used by SpotForecast2",
    )
    parser.add_argument(
        "--use-exogenous-features",
        type=str,
        default="true",
        help="Whether to let SpotForecast2 build exogenous features",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="DateTime",
        help="Datetime column name used by SpotForecast2",
    )
    parser.add_argument("--latitude", type=float, default=51.5, help="Latitude")
    parser.add_argument("--longitude", type=float, default=10.5, help="Longitude")
    parser.add_argument("--timezone", type=str, default="Europe/Berlin", help="Timezone")
    parser.add_argument("--country-code", type=str, default="DE", help="Country code")
    parser.add_argument("--state", type=str, default="NW", help="State code")
    parser.add_argument(
        "--show",
        type=str,
        default="false",
        help="Whether to display Optuna figures",
    )
    parser.add_argument(
        "--candidate-models",
        type=str,
        default="LightGBM,XGBoost,Ridge,CatBoost",
        help=(
            "Comma-separated list of model types to compare during model selection. "
            "Set to 'LightGBM' to skip comparison and use LightGBM only. "
            "Supported: LightGBM, XGBoost, Ridge, CatBoost."
        ),
    )
    parser.add_argument(
        "--n-trials-compare",
        type=int,
        default=5,
        help="Optuna trials per model per target during model selection (smaller = faster)",
    )
    parser.add_argument(
        "--compare-metric",
        type=str,
        default="mean_absolute_error",
        help="Metric used to rank models during model selection (lower is better)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def _write_tuned_config(
    cfg: dict[str, Any],
    config_path: Path,
    tuned_yaml_paths: dict[str, Path],
    task_name: str,
) -> Path:
    """Write a new config file with channel_config_files pointing to tuned YAMLs."""
    import copy

    tuned_cfg = copy.deepcopy(cfg)
    tuned_cfg.setdefault("train", {})["channel_config_files"] = {
        panel_id: str(yaml_path)
        for panel_id, yaml_path in tuned_yaml_paths.items()
    }

    out_path = config_path.parent / f"config_{task_name}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned_cfg, f, sort_keys=False)
    return out_path


def main() -> int:
    args = build_parser().parse_args()
    args.show = _parse_bool(str(args.show))
    args.use_exogenous_features = _parse_bool(str(args.use_exogenous_features))

    cfg = load_config(Path(args.config))
    logger = get_logger("SpotForecast2Tuning")
    dm = DataManager(cfg, logger)

    panel_data = dm.load_processed_data()
    panel_ids = [args.panel_id] if args.panel_id else cfg.get("panels", {}).get("panel_ids", [])

    if not panel_ids:
        raise RuntimeError("No panel ids found in config")

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    run_ts = generate_timestamp()

    logger.info("Starting tuning run %s", run_ts)
    logger.info("Panels: %s", ", ".join(panel_ids))

    tuned_yaml_paths: dict[str, Path] = {}
    for panel_id in panel_ids:
        if panel_id not in panel_data:
            logger.warning("Skipping panel %s: no processed data found", panel_id)
            continue
        yaml_path, _ = tune_panel(
            panel_id=panel_id,
            panel_df=panel_data[panel_id],
            cfg=cfg,
            args=args,
            output_dir=output_dir,
            results_dir=results_dir,
            logger=logger,
        )
        tuned_yaml_paths[panel_id] = yaml_path

    if tuned_yaml_paths:
        task_name = args.task
        config_path = Path(args.config)
        tuned_config_path = _write_tuned_config(cfg, config_path, tuned_yaml_paths, task_name)
        logger.info("Wrote tuned config to %s", tuned_config_path)

    logger.info("Tuning run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
