"""Per-panel training entry point: ``SpotforecastTrainer``.

Trains one ``ForecasterRecursive`` per target channel for a panel, with the
spotanomaly2-specific touches that the bare spotforecast2 ``MultiTask`` path
doesn't cover: per-channel model selection from YAML, imputation-flag-aware
sample weighting, optional Ridge-residual anomaly pre-pass to mask training
regions, and one-step-ahead test eval that mirrors what detection / live mode
will see at scoring time.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from spotanomaly2.application.config import load_panel_channel_config
from spotanomaly2.infrastructure import logging, storage
from spotanomaly2.infrastructure.storage import generate_timestamp

from .factory import _create_forecaster
from .prediction import _difference, _predict_one_step_integrated
from .preprocessing import (
    _build_strict_training_sample_mask,
    _compute_observed_mask,
    _detect_anomalies_via_ridge,
    _ensure_freq,
    _interpolate_inplace,
    _mask_known_anomalies,
    _split_panel_columns,
    _time_series_train_test_split,
)


class SpotforecastTrainer:
    """Trains one ``ForecasterRecursive`` per target channel for a panel."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("SpotforecastTrainer")

    def _get_weight_suffix(self) -> str:
        return self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")

    def train_panel(
        self,
        panel_id: str,
        panel_data: pd.DataFrame,
        timestamp: str | None = None,
        save_model: bool = True,
    ) -> tuple[pd.DataFrame, str]:
        """Train one ForecasterRecursive per target channel for a panel."""
        base_models_dir = Path(self.config["paths"]["models_dir"])
        train_ratio = self.config["train"]["train_ratio"]
        model_label = self.config["train"]["fallback_model"]
        n_lags = self.config["train"].get("lags", 24)
        random_seed = self.config["train"].get("random_seed", 42)
        # Predict Δy[t] = y[t] - y[t-1] instead of y[t]. At inference we
        # integrate one-step: y_pred[t] = y[t-1] + Δy_pred[t]. Makes the
        # target stationary so no model — degenerate or not — can win by
        # collapsing to the training mean. Set 0 to disable.
        diff_order = int(self.config.get("train", {}).get("differentiation", 1))

        if timestamp is None:
            timestamp = generate_timestamp()
        models_dir = base_models_dir / timestamp

        self.logger.info(f"Training spotforecast2 models on {len(panel_data)} rows for panel {panel_id}")

        configured_exog_columns = self.config["train"].get("exog_columns", [])

        weight_suffix = self._get_weight_suffix()
        weight_residuals_enabled = self.config.get("residual_weighting", {}).get("enabled", False)
        if weight_residuals_enabled:
            self.logger.info("residual_weighting enabled: exogenous columns excluded from model features")
        target_cols, exog_columns = _split_panel_columns(
            panel_data, configured_exog_columns, weight_suffix, weight_residuals_enabled
        )

        known_anomalies = self.config.get("known_anomalies", [])
        known_anomaly_buffer = self.config["train"].get("known_anomaly_buffer")
        if known_anomalies and known_anomaly_buffer:
            panel_data = _mask_known_anomalies(panel_data, known_anomalies, known_anomaly_buffer, columns=target_cols)

        train_df, test_df = _time_series_train_test_split(panel_data, train_ratio=train_ratio)

        fallback_freq = self.config.get("process", {}).get("resample", {}).get("freq", "5min")
        train_df = _ensure_freq(train_df, fallback_freq)
        test_df = _ensure_freq(test_df, fallback_freq)

        self.logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        panel_cfg_from_yaml = load_panel_channel_config(panel_id, self.config)

        panel_default_section = panel_cfg_from_yaml.get("default")
        if isinstance(panel_default_section, dict):
            panel_default_model = panel_default_section.get("model")
            panel_default_params = panel_default_section.get("params", {})
        else:
            panel_default_model = panel_cfg_from_yaml.get("default_model")
            panel_default_params = panel_cfg_from_yaml.get("default_params", {})
        channel_cfg_map = panel_cfg_from_yaml.get("channels", {})

        forecasters: dict[str, Any] = {}
        channel_model_specs: dict[str, dict[str, Any]] = {}
        y_pred_test = np.zeros((len(test_df), len(target_cols)))

        for i, target_col in enumerate(target_cols):
            channel_cfg = channel_cfg_map.get(target_col, {})
            if not isinstance(channel_cfg, dict):
                channel_cfg = {}

            channel_model_name = channel_cfg.get("model") or panel_default_model or model_label
            default_name_lc = (panel_default_model or model_label or "").strip().lower()
            channel_model_params: dict[str, Any] = {}
            if (channel_model_name or "").strip().lower() == default_name_lc:
                channel_model_params.update(panel_default_params or {})
            channel_model_params.update(channel_cfg.get("params", {}) or {})

            self.logger.info(f"  Training forecaster for: {target_col} (model={channel_model_name})")

            # Resolve per-channel lag spec from the panel YAML (best_lags), falling
            # back to the global train.lags. effective_lags is what we set on the
            # forecaster; effective_n_lags is the int used for sufficiency checks.
            effective_lags: Any = channel_cfg.get("best_lags", n_lags)
            if isinstance(effective_lags, (list, tuple)) and effective_lags:
                effective_n_lags = int(max(effective_lags))
                effective_lags = list(effective_lags)
            else:
                try:
                    effective_n_lags = int(effective_lags)
                    effective_lags = effective_n_lags
                except (TypeError, ValueError):
                    effective_n_lags = n_lags
                    effective_lags = n_lags

            y_train_raw = train_df[target_col].copy()
            y_train_raw.name = target_col

            observed_mask = _compute_observed_mask(train_df, target_col, weight_suffix)
            y_train = y_train_raw.copy()
            y_train.loc[~observed_mask] = np.nan
            if y_train.isna().any():
                y_train = _interpolate_inplace(y_train)
            if len(y_train) < effective_n_lags + 10:
                self.logger.warning(f"  Skipping {target_col}: insufficient data ({len(y_train)} rows)")
                continue

            sample_mask_for_weights = None
            if self.config.get("train", {}).get("exclude_imputed_training_samples", False):
                sample_mask_for_weights = _build_strict_training_sample_mask(
                    observed_mask=observed_mask,
                    n_lags=effective_n_lags,
                )
                potential_rows = max(len(y_train) - effective_n_lags, 0)
                kept_rows = int(sample_mask_for_weights.iloc[effective_n_lags:].sum())
                self.logger.info(
                    f"    {target_col}: excluding {potential_rows - kept_rows} training sample(s) "
                    f"that contain imputed target/lag values"
                )
                if kept_rows == 0:
                    self.logger.warning(f"  Skipping {target_col}: no fully observed training samples left")
                    continue

            train_cfg = self.config.get("train", {})
            if train_cfg.get("auto_clean_anomalies", False):
                anomaly_mask = _detect_anomalies_via_ridge(
                    y_train,
                    n_lags=effective_n_lags,
                    threshold_scale=train_cfg.get("auto_clean_threshold", 4.0),
                    buffer=train_cfg.get("auto_clean_buffer", 3),
                )
                n_flagged = int(anomaly_mask.sum())
                if n_flagged > 0:
                    self.logger.info(f"    {target_col}: auto-cleaning flagged {n_flagged} suspected anomaly points")
                    if sample_mask_for_weights is not None:
                        sample_mask_for_weights = sample_mask_for_weights & ~anomaly_mask
                    else:
                        sample_mask_for_weights = ~anomaly_mask

            exog_train = train_df[exog_columns].loc[y_train.index] if exog_columns else None

            forecaster = _create_forecaster(
                channel_model_name,
                channel_model_params,
                n_lags=effective_lags,
                random_seed=random_seed,
                has_exog=exog_train is not None,
                logger=self.logger,
            )

            if sample_mask_for_weights is not None:

                def _weight_func(index, mask=sample_mask_for_weights):
                    return mask.reindex(index).fillna(False).astype(float).to_numpy()

                forecaster.weight_func = _weight_func

            # Fit on Δy when differentiation > 0. spotforecast2's
            # transformer_y / weight_func still apply normally to the
            # differenced target.
            if diff_order > 0:
                y_for_fit = _difference(y_train, diff_order).dropna()
                exog_for_fit = exog_train.loc[y_for_fit.index] if exog_train is not None else None
            else:
                y_for_fit = y_train
                exog_for_fit = exog_train
            forecaster.fit(y=y_for_fit, exog=exog_for_fit)

            # Detach the closure so joblib can serialise the fitted forecaster.
            if sample_mask_for_weights is not None:
                forecaster.weight_func = None
                forecaster.source_code_weight_func = None

            channel_model_specs[target_col] = {
                "model": channel_model_name,
                "params": channel_model_params,
                "lags": effective_lags,
            }

            # One-step-ahead test predictions on real observed lags (mirrors
            # what detection / live mode does — see _predict_one_step_integrated).
            test_observed = _compute_observed_mask(test_df, target_col, weight_suffix)
            y_test_for_lags = test_df[target_col].copy()
            y_test_for_lags.loc[~test_observed] = np.nan
            if y_test_for_lags.isna().any():
                y_test_for_lags = _interpolate_inplace(y_test_for_lags)

            full_y_raw = pd.concat([y_train, y_test_for_lags])
            full_y_raw.name = target_col

            full_exog = None
            if exog_columns:
                full_exog = pd.concat([train_df[exog_columns], test_df[exog_columns]])
                if full_exog.isna().any().any():
                    full_exog = _interpolate_inplace(full_exog)
                full_exog = full_exog.loc[full_y_raw.index]

            try:
                y_pred_test[:, i] = _predict_one_step_integrated(
                    forecaster, full_y_raw, full_exog, test_df.index, diff_order
                )
            except Exception as exc:
                self.logger.warning(f"  One-step-ahead test prediction failed for {target_col}: {exc}")
                y_pred_test[:, i] = np.nan

            forecasters[target_col] = forecaster

            y_test_col = test_df[target_col].values
            valid = ~np.isnan(y_test_col)
            if valid.any():
                rmse = np.sqrt(np.mean((y_test_col[valid] - y_pred_test[valid, i]) ** 2))
                mae = np.mean(np.abs(y_test_col[valid] - y_pred_test[valid, i]))
                self.logger.info(f"    {target_col}: RMSE={rmse:.4f}, MAE={mae:.4f}")

        self.logger.info(f"Trained {len(forecasters)} forecasters")

        y_test_vals = test_df[target_cols].values
        rmse_results = np.sqrt(np.nanmean((y_test_vals - y_pred_test) ** 2, axis=0))
        mae_results = np.nanmean(np.abs(y_test_vals - y_pred_test), axis=0)

        res_df = pd.DataFrame({"rmse": rmse_results, "mae": mae_results}, index=target_cols)

        unique_channel_models = {spec["model"] for spec in channel_model_specs.values()}
        if len(unique_channel_models) == 1:
            actual_model_label = next(iter(unique_channel_models))
        elif len(unique_channel_models) > 1:
            actual_model_label = "Multi"
        else:
            actual_model_label = model_label

        model_data = {
            "forecasters": forecasters,
            "n_lags": n_lags,
            "target_cols": target_cols,
            "exog_columns": exog_columns,
            "channel_models": channel_model_specs,
            "differentiation": diff_order,
            "model_type": f"spotforecast2_{actual_model_label.lower()}",
            "model_name": actual_model_label,
            "train_ratio": train_ratio,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_start_timestamp": (
                train_df.index.min().isoformat()
                if len(train_df) > 0 and isinstance(train_df.index, pd.DatetimeIndex)
                else None
            ),
            "train_end_timestamp": (
                train_df.index.max().isoformat()
                if len(train_df) > 0 and isinstance(train_df.index, pd.DatetimeIndex)
                else None
            ),
            "test_start_timestamp": (
                test_df.index.min().isoformat()
                if len(test_df) > 0 and isinstance(test_df.index, pd.DatetimeIndex)
                else None
            ),
            "timestamp": timestamp,
        }

        if save_model:
            storage.ensure_dir(models_dir)
            model_path = models_dir / f"fc_model_panel_{panel_id}.pkl"
            joblib.dump(model_data, model_path)
            self.logger.info(f"Saved spotforecast2 model to {model_path}")

        self.logger.info(f"Average RMSE: {rmse_results.mean():.4f}")
        return res_df, timestamp

    def predict(
        self,
        model_data: dict[str, Any],
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Batch one-step-ahead predictions over ``df`` for anomaly scoring.

        Builds the lag/window-feature matrix from the *actually observed* values
        at every timestep (via ``forecaster.create_train_X_y``) and scores them
        in one pass, so each prediction is conditioned on real data — not on
        prior predictions, which is what the recursive ``predict`` would do.
        """
        forecasters = model_data["forecasters"]
        target_cols = model_data["target_cols"]
        exog_columns = model_data.get("exog_columns", [])
        # Older model artifacts (pre-differentiation) default to 0 = predict
        # raw y. Newer artifacts carry the actual order used at train time.
        diff_order = int(model_data.get("differentiation", 0))
        weight_suffix = self._get_weight_suffix()

        df = _ensure_freq(df)
        if history_df is not None:
            history_df = _ensure_freq(history_df)

        predictions: dict[str, np.ndarray] = {}

        for target_col in target_cols:
            forecaster = forecasters.get(target_col)
            if forecaster is None:
                predictions[target_col] = np.full(len(df), np.nan)
                continue

            if history_df is not None:
                full_y = pd.concat([history_df[target_col], df[target_col]])
                observed_mask = pd.concat(
                    [
                        _compute_observed_mask(history_df, target_col, weight_suffix),
                        _compute_observed_mask(df, target_col, weight_suffix),
                    ]
                )
            else:
                full_y = df[target_col].copy()
                observed_mask = _compute_observed_mask(df, target_col, weight_suffix)
            full_y.name = target_col

            full_y = full_y.copy()
            full_y.loc[~observed_mask] = np.nan
            if full_y.isna().any():
                full_y = _interpolate_inplace(full_y)
            if len(full_y) == 0:
                predictions[target_col] = np.full(len(df), np.nan)
                continue

            full_y = _ensure_freq(full_y)

            exog_full = None
            if exog_columns:
                cols_present = [c for c in exog_columns if c in df.columns]
                if cols_present:
                    if history_df is not None:
                        exog_full = pd.concat([history_df[cols_present], df[cols_present]])
                    else:
                        exog_full = df[cols_present].copy()
                    exog_full = exog_full.loc[full_y.index]
                    # Interpolate NaN the same way we do for y. With a
                    # transformer_exog in place, NaN cells would otherwise
                    # propagate through StandardScaler.transform and crash
                    # estimator.predict on linear models.
                    if exog_full.isna().any().any():
                        exog_full = _interpolate_inplace(exog_full)

            try:
                predictions[target_col] = _predict_one_step_integrated(
                    forecaster, full_y, exog_full, df.index, diff_order
                )
            except Exception as e:
                self.logger.warning(f"Prediction failed for {target_col}: {e}. Using NaN.")
                predictions[target_col] = np.full(len(df), np.nan)

        return pd.DataFrame(predictions, index=df.index)

    def train_all_panels(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        training_timestamp = generate_timestamp()
        return {pid: self.train_panel(pid, df, timestamp=training_timestamp) for pid, df in panel_data.items()}

    def run(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        self.logger.info("Starting spotforecast2 model training...")
        results = self.train_all_panels(panel_data)
        self.logger.info("spotforecast2 model training completed")
        return results
