"""Per-panel training entry point: ``SpotforecastTrainer``.

Trains one ``ForecasterRecursive`` per target channel for a panel, with the
spotanomaly2-specific touches that the bare spotforecast2 ``MultiTask`` path
doesn't cover: per-channel model selection from YAML, imputation-flag-aware
sample weighting, optional Ridge-residual anomaly pre-pass to mask training
regions, and one-step-ahead test eval that mirrors what detection / live mode
will see at scoring time.

Inference against a trained model lives in ``SpotforecastPredictor``.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from spotanomaly2.application.config import load_panel_channel_config, resolve_data_split
from spotanomaly2.infrastructure import logging, storage
from spotanomaly2.infrastructure.storage import generate_timestamp

from .channel_prep import (
    attach_weight_func,
    impute_exog,
    prepare_channel,
    prepare_panel,
    resolve_train_settings,
)
from .factory import _create_forecaster
from .prediction import _predict_one_step_integrated
from .preprocessing import _compute_observed_mask


class SpotforecastTrainer:
    """Trains one ``ForecasterRecursive`` per target channel for a panel."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("SpotforecastTrainer")

    def train_panel(
        self,
        panel_id: str,
        panel_data: pd.DataFrame,
        timestamp: str | None = None,
        save_model: bool = True,
    ) -> tuple[pd.DataFrame, str]:
        """Train one ForecasterRecursive per target channel for a panel."""
        knobs = resolve_train_settings(self.config)
        split = resolve_data_split(self.config)
        model_label = self.config["train"]["fallback_model"]
        fallback_lags = knobs.fallback_lags
        random_seed = knobs.random_seed
        diff_order = knobs.diff_order
        weight_suffix = knobs.weight_suffix

        if timestamp is None:
            timestamp = generate_timestamp()

        self.logger.info(f"Training spotforecast2 models on {len(panel_data)} rows for panel {panel_id}")

        panel_data, target_cols, exog_columns = prepare_panel(self.config, panel_data, weight_suffix, self.logger)
        train_df, val_df, test_df, test_start_timestamp = self._split_train_val_test(panel_data, split)
        panel_default_model, panel_default_params, channel_cfg_map = self._load_and_resolve_panel_config(panel_id)
        full_exog = self._build_full_exog(train_df, val_df, test_df, exog_columns)

        forecasters: dict[str, Any] = {}
        channel_model_specs: dict[str, dict[str, Any]] = {}
        y_pred_val = np.zeros((len(val_df), len(target_cols)))
        y_pred_test = np.zeros((len(test_df), len(target_cols)))

        for i, target_col in enumerate(target_cols):
            channel_cfg = channel_cfg_map.get(target_col, {})
            if not isinstance(channel_cfg, dict):
                channel_cfg = {}

            channel_result = self._train_channel(
                target_col=target_col,
                channel_cfg=channel_cfg,
                panel_default_model=panel_default_model,
                panel_default_params=panel_default_params,
                model_label=model_label,
                fallback_lags=fallback_lags,
                random_seed=random_seed,
                diff_order=diff_order,
                weight_suffix=weight_suffix,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                exog_columns=exog_columns,
                full_exog=full_exog,
            )
            if channel_result is None:
                continue
            forecaster, model_spec, val_preds, test_preds = channel_result
            forecasters[target_col] = forecaster
            channel_model_specs[target_col] = model_spec
            y_pred_val[:, i] = val_preds
            if len(test_df) > 0:
                y_pred_test[:, i] = test_preds

        self.logger.info(f"Trained {len(forecasters)} forecasters")

        val_metrics = self._compute_eval_metrics(val_df, target_cols, y_pred_val, weight_suffix)
        res_df = val_metrics.rename(columns={"rmse": "val_rmse", "mae": "val_mae"})
        self.logger.info(f"Average val RMSE: {res_df['val_rmse'].mean():.4f} (tuning fold — optimistic)")
        self._append_test_metrics(res_df, test_df, target_cols, y_pred_test, weight_suffix)

        if save_model:
            model_data = self._build_model_artifact(
                forecasters=forecasters,
                target_cols=target_cols,
                exog_columns=exog_columns,
                channel_model_specs=channel_model_specs,
                diff_order=diff_order,
                n_lags=fallback_lags,
                model_label=model_label,
                split=split,
                train_df=train_df,
                val_df=val_df,
                test_start_timestamp=test_start_timestamp,
                timestamp=timestamp,
            )

            self._save_model_artifact(model_data, panel_id, timestamp)

        return res_df, timestamp

    def _split_train_val_test(
        self,
        panel_data: pd.DataFrame,
        split,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
        """Carve panel_data into (train_df, val_df, test_df, test_start_timestamp).

        ``train_df`` is the forecaster's fit window; ``val_df`` is the
        tuning-fold eval window (mirrors the tuner's CV validation window, so
        its metric is optimistically biased by hyperparameter selection).
        ``test_df`` is the final slice the trainer and tuner never *fit* on:
        it is returned here only for a **read-only**, tuning-independent eval
        metric — predicting on it does not refit anything, so the scorer's
        leakage-free residuals at detect time are unaffected. Its start is
        persisted as ``test_start_timestamp`` so the detector can draw the
        seen/unseen boundary.
        """
        n = len(panel_data)
        train_end = int(n * split.train / 100)
        test_start = int(n * (split.train + split.val) / 100)
        train_df = panel_data.iloc[:train_end]
        val_df = panel_data.iloc[train_end:test_start]
        test_df = panel_data.iloc[test_start:]
        test_start_ts: str | None = None
        if isinstance(panel_data.index, pd.DatetimeIndex) and test_start < n:
            test_start_ts = panel_data.index[test_start].isoformat()
        self.logger.info(
            f"Train size: {len(train_df)} ({split.train}%), Val size: {len(val_df)} ({split.val}%), "
            f"Test size: {len(test_df)} ({split.test}%, eval-only — never fit on)"
        )
        return train_df, val_df, test_df, test_start_ts

    def _load_and_resolve_panel_config(self, panel_id: str) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
        """Load per-panel YAML and extract (default_model, default_params, channel_cfg_map)."""
        panel_cfg = load_panel_channel_config(panel_id, self.config)

        default_section = panel_cfg.get("default")
        if isinstance(default_section, dict):
            panel_default_model = default_section.get("model")
            panel_default_params = default_section.get("params", {})
        else:
            panel_default_model = panel_cfg.get("default_model")
            panel_default_params = panel_cfg.get("default_params", {})
        channel_cfg_map = panel_cfg.get("channels", {})
        return panel_default_model, panel_default_params, channel_cfg_map

    @staticmethod
    def _resolve_channel_model_spec(
        channel_cfg: dict[str, Any],
        panel_default_model: str | None,
        panel_default_params: dict[str, Any],
        model_label: str,
    ) -> tuple[str, dict[str, Any]]:
        """Resolve (model_name, model_params) for one channel.

        Precedence: channel YAML > panel default > global fallback. Panel default
        params apply only when the channel inherits the panel's model name.
        """
        channel_model_name = channel_cfg.get("model") or panel_default_model or model_label
        default_name_lc = (panel_default_model or model_label or "").strip().lower()
        channel_model_params: dict[str, Any] = {}
        if (channel_model_name or "").strip().lower() == default_name_lc:
            channel_model_params.update(panel_default_params or {})
        channel_model_params.update(channel_cfg.get("params", {}) or {})
        return channel_model_name, channel_model_params

    @staticmethod
    def _resolve_channel_lags(
        channel_cfg: dict[str, Any],
        fallback_lags: int | list[int],
    ) -> tuple[int | list[int], int]:
        """Return (effective_lags, effective_n_lags) from per-channel ``best_lags`` if any.

        ``effective_lags`` is what we set on the forecaster (may be a list of
        specific lag offsets). ``effective_n_lags`` is the int used for
        sufficiency checks (max lag = longest history needed).
        """
        effective = channel_cfg.get("best_lags", fallback_lags)
        if isinstance(effective, (list, tuple)) and effective:
            return list(effective), int(max(effective))
        try:
            n = int(effective)
            return n, n
        except (TypeError, ValueError):
            if isinstance(fallback_lags, (list, tuple)) and fallback_lags:
                return list(fallback_lags), int(max(fallback_lags))
            return fallback_lags, int(fallback_lags) if not isinstance(fallback_lags, (list, tuple)) else 24

    def _fit_channel_forecaster(
        self,
        model_name: str,
        model_params: dict[str, Any],
        effective_lags: int | list[int],
        random_seed: int,
        y_fit: pd.Series,
        exog_fit: pd.DataFrame | None,
        sample_mask: pd.Series | None,
    ):
        """Create forecaster, attach weight_func, fit, detach for serialization.

        ``y_fit`` / ``exog_fit`` arrive already differenced + aligned from
        ``prepare_channel``; spotforecast2's transformer_y / weight_func still
        apply normally to the (possibly differenced) target.
        """
        forecaster = _create_forecaster(
            model_name,
            model_params,
            n_lags=effective_lags,
            random_seed=random_seed,
            has_exog=exog_fit is not None,
            logger=self.logger,
        )

        attach_weight_func(forecaster, sample_mask)
        forecaster.fit(y=y_fit, exog=exog_fit)

        # Detach the closure so joblib can serialise the fitted forecaster.
        if sample_mask is not None:
            forecaster.weight_func = None
            forecaster.source_code_weight_func = None

        return forecaster

    def _build_full_exog(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        exog_columns: list[str],
    ) -> pd.DataFrame | None:
        """Concat + impute the train+val+test exog frame once for the panel.

        The result is reused across every channel's one-step-ahead val and
        test prediction. Exog columns don't depend on the target, so rebuilding
        this per channel is pure waste. Gaps are filled with the config's
        imputation method so train sees the same exog features tune does.
        """
        if not exog_columns:
            return None
        full_exog = pd.concat([train_df[exog_columns], val_df[exog_columns], test_df[exog_columns]])
        return impute_exog(self.config, full_exog, exog_columns)

    def _predict_eval_window(
        self,
        forecaster,
        history_y: pd.Series,
        eval_df: pd.DataFrame,
        target_col: str,
        full_exog: pd.DataFrame | None,
        diff_order: int,
    ) -> np.ndarray:
        """One-step-ahead predictions over ``eval_df`` on real observed lags.

        ``history_y`` is the raw target *preceding* ``eval_df`` — the train
        window for the val metric, train+val for the test metric. Mirrors
        what detection / live mode does at scoring time (see
        ``_predict_one_step_integrated``), so the train-time eval metric is the
        same one production reports.
        """
        # Targets are already gap-free (process stage + _apply_known_anomaly_imputation).
        y_eval_for_lags = eval_df[target_col].copy()

        full_y_raw = pd.concat([history_y, y_eval_for_lags])
        full_y_raw.name = target_col

        exog_for_pred = full_exog.loc[full_y_raw.index] if full_exog is not None else None

        try:
            return _predict_one_step_integrated(forecaster, full_y_raw, exog_for_pred, eval_df.index, diff_order)
        except Exception as exc:
            self.logger.warning(f"  One-step-ahead prediction failed for {target_col}: {exc}")
            return np.full(len(eval_df), np.nan)

    def _train_channel(
        self,
        target_col: str,
        channel_cfg: dict[str, Any],
        panel_default_model: str | None,
        panel_default_params: dict[str, Any],
        model_label: str,
        fallback_lags: int | list[int],
        random_seed: int,
        diff_order: int,
        weight_suffix: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        exog_columns: list[str],
        full_exog: pd.DataFrame | None,
    ) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray] | None:
        """Train one forecaster + run val/test eval for a single channel.

        Returns ``(forecaster, model_spec, val_predictions, test_predictions)``
        or ``None`` when the channel is skipped (insufficient data or all
        training samples masked out). ``test_predictions`` is empty when there
        is no test split.
        """
        channel_model_name, channel_model_params = self._resolve_channel_model_spec(
            channel_cfg, panel_default_model, panel_default_params, model_label
        )
        effective_lags, effective_n_lags = self._resolve_channel_lags(channel_cfg, fallback_lags)

        self.logger.info(f"  Training forecaster for: {target_col} (model={channel_model_name})")

        # Shared per-channel setup (target/exog split is already done; this does
        # the imputation-flag mask, exog imputation, and optional Δy) so train
        # and tune fit the exact same inputs. ``None`` => skip this channel.
        channel = prepare_channel(
            self.config,
            train_df,
            target_col,
            exog_columns,
            weight_suffix,
            n_lags_for_mask=effective_n_lags,
            diff_order=diff_order,
            logger=self.logger,
        )
        if channel is None:
            return None

        forecaster = self._fit_channel_forecaster(
            channel_model_name,
            channel_model_params,
            effective_lags,
            random_seed,
            channel.y_fit,
            channel.exog_fit,
            channel.sample_mask,
        )

        # Val window: lag context is the train window (channel.y_raw).
        val_preds = self._predict_eval_window(forecaster, channel.y_raw, val_df, target_col, full_exog, diff_order)

        # Test window: lag context is train+val (everything the forecaster's
        # predictions can legitimately condition on before the test slice).
        # Read-only — the forecaster is already fit and is not touched here.
        if len(test_df) > 0:
            test_history = pd.concat([channel.y_raw, val_df[target_col]])
            test_preds = self._predict_eval_window(forecaster, test_history, test_df, target_col, full_exog, diff_order)
        else:
            test_preds = np.empty(0)

        # Score against real observations only — rows where __weight < 0.5
        # (process-imputed or known-anomaly) are interpolated values and would
        # bias RMSE/MAE if compared against the model's prediction.
        val_observed = _compute_observed_mask(val_df, target_col, weight_suffix).to_numpy()
        if val_observed.any():
            y_real = val_df[target_col].to_numpy(dtype=float)[val_observed]
            preds_real = val_preds[val_observed]
            rmse = np.sqrt(np.mean((y_real - preds_real) ** 2))
            mae = np.mean(np.abs(y_real - preds_real))
            self.logger.info(f"    {target_col}: RMSE={rmse:.4f}, MAE={mae:.4f}")

        return (
            forecaster,
            {
                "model": channel_model_name,
                "params": channel_model_params,
                "lags": effective_lags,
            },
            val_preds,
            test_preds,
        )

    @staticmethod
    def _compute_eval_metrics(
        eval_df: pd.DataFrame,
        target_cols: list[str],
        y_pred: np.ndarray,
        weight_suffix: str,
    ) -> pd.DataFrame:
        """Build a per-channel ``rmse``/``mae`` DataFrame from eval predictions.

        Imputed / known-anomaly rows (``__weight < 0.5``) are excluded from the
        metric — their stored value is interpolated, not observed, so comparing
        it to the model's prediction would bias the score.
        """
        y_eval_vals = eval_df[target_cols].to_numpy(dtype=float).copy()
        for i, col in enumerate(target_cols):
            obs = _compute_observed_mask(eval_df, col, weight_suffix).to_numpy()
            y_eval_vals[~obs, i] = np.nan
        rmse = np.sqrt(np.nanmean((y_eval_vals - y_pred) ** 2, axis=0))
        mae = np.nanmean(np.abs(y_eval_vals - y_pred), axis=0)
        return pd.DataFrame({"rmse": rmse, "mae": mae}, index=target_cols)

    def _append_test_metrics(
        self,
        res_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_cols: list[str],
        y_pred_test: np.ndarray,
        weight_suffix: str,
    ) -> None:
        """Add ``test_rmse``/``test_mae`` columns to ``res_df`` in place.

        These are the per-channel forecast errors on the held-out ``test``
        split, which (unlike ``val_rmse``/``val_mae`` on the tuning fold) the
        hyperparameter search never selected against — the unbiased baseline
        for production monitoring. Columns are always added (NaN when there is
        no test split) so the eval CSV schema is stable.
        """
        if len(test_df) > 0:
            test_metrics = self._compute_eval_metrics(test_df, target_cols, y_pred_test, weight_suffix)
            res_df["test_rmse"] = test_metrics["rmse"]
            res_df["test_mae"] = test_metrics["mae"]
            self.logger.info(
                f"Average test RMSE: {test_metrics['rmse'].mean():.4f} "
                "(held-out, tuning-independent — monitoring baseline)"
            )
        else:
            res_df["test_rmse"] = np.nan
            res_df["test_mae"] = np.nan
            self.logger.warning("No test split available; test-window eval metric skipped.")

    def _build_model_artifact(
        self,
        *,
        forecasters: dict[str, Any],
        target_cols: list[str],
        exog_columns: list[str],
        channel_model_specs: dict[str, dict[str, Any]],
        diff_order: int,
        n_lags: int | list[int],
        model_label: str,
        split,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_start_timestamp: str | None,
        timestamp: str,
    ) -> dict[str, Any]:
        """Package the dict that ``SpotforecastPredictor`` and the detector consume.

        ``test_start_timestamp`` marks the first row of the held-out test split
        (never fit on by trainer or tuner). The detector uses it to draw the
        leakage-free seen/unseen boundary.
        """
        unique_models = {spec["model"] for spec in channel_model_specs.values()}
        if len(unique_models) == 1:
            actual_label = next(iter(unique_models))
        elif len(unique_models) > 1:
            actual_label = "Multi"
        else:
            actual_label = model_label

        return {
            "forecasters": forecasters,
            "n_lags": n_lags,
            "target_cols": target_cols,
            "exog_columns": exog_columns,
            "channel_models": channel_model_specs,
            "differentiation": diff_order,
            "model_type": f"spotforecast2_{actual_label.lower()}",
            "model_name": actual_label,
            "split": {"train": split.train, "val": split.val, "test": split.test},
            "train_size": len(train_df),
            "val_size": len(val_df),
            "train_start_timestamp": self._iso_min(train_df),
            "train_end_timestamp": self._iso_max(train_df),
            "val_start_timestamp": self._iso_min(val_df),
            "val_end_timestamp": self._iso_max(val_df),
            "test_start_timestamp": test_start_timestamp,
            "timestamp": timestamp,
        }

    @staticmethod
    def _iso_min(df: pd.DataFrame) -> str | None:
        if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex):
            return df.index.min().isoformat()
        return None

    @staticmethod
    def _iso_max(df: pd.DataFrame) -> str | None:
        if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex):
            return df.index.max().isoformat()
        return None

    def _save_model_artifact(self, model_data: dict[str, Any], panel_id: str, timestamp: str) -> None:
        """Persist the trained model dict to ``models_dir/<timestamp>/fc_model_panel_<id>.pkl``."""
        models_dir = Path(self.config["paths"]["models_dir"]) / timestamp
        storage.ensure_dir(models_dir)
        model_path = models_dir / f"fc_model_panel_{panel_id}.pkl"
        joblib.dump(model_data, model_path)
        self.logger.info(f"Saved spotforecast2 model to {model_path}")

    def train_all_panels(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        training_timestamp = generate_timestamp()
        return {pid: self.train_panel(pid, df, timestamp=training_timestamp) for pid, df in panel_data.items()}

    def run(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        self.logger.info("Starting spotforecast2 model training...")
        results = self.train_all_panels(panel_data)
        self.logger.info("spotforecast2 model training completed")
        return results
