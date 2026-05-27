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

from spotanomaly2.application.config import load_panel_channel_config
from spotanomaly2.infrastructure import logging, storage
from spotanomaly2.infrastructure.storage import generate_timestamp

from .factory import _create_forecaster
from .panel_layout import _split_panel_columns
from .prediction import _difference, _predict_one_step_integrated
from .preprocessing import (
    _build_strict_training_sample_mask,
    _compute_observed_mask,
    _detect_anomalies_via_ridge,
    _ensure_freq,
    _interpolate_inplace,
    _mask_known_anomalies,
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
        train_ratio = self.config["train"]["train_ratio"]
        model_label = self.config["train"]["fallback_model"]
        n_lags = self.config["train"].get("lags", 24)
        random_seed = self.config["train"].get("random_seed", 42)
        # Predict Δy[t] = y[t] - y[t-1] instead of y[t]. At inference we
        # integrate one-step: y_pred[t] = y[t-1] + Δy_pred[t]. Makes the
        # target stationary so no model — degenerate or not — can win by
        # collapsing to the training mean. Set 0 to disable.
        diff_order = int(self.config.get("train", {}).get("differentiation", 1))
        weight_suffix = self._get_weight_suffix()

        if timestamp is None:
            timestamp = generate_timestamp()

        self.logger.info(f"Training spotforecast2 models on {len(panel_data)} rows for panel {panel_id}")

        panel_data, target_cols, exog_columns = self._prepare_panel_data(panel_data, weight_suffix)
        train_df, test_df = self._split_train_test(panel_data, train_ratio)
        panel_default_model, panel_default_params, channel_cfg_map = self._load_and_resolve_panel_config(panel_id)

        forecasters: dict[str, Any] = {}
        channel_model_specs: dict[str, dict[str, Any]] = {}
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
                n_lags=n_lags,
                random_seed=random_seed,
                diff_order=diff_order,
                weight_suffix=weight_suffix,
                train_df=train_df,
                test_df=test_df,
                exog_columns=exog_columns,
            )
            if channel_result is None:
                continue
            forecaster, model_spec, channel_preds = channel_result
            forecasters[target_col] = forecaster
            channel_model_specs[target_col] = model_spec
            y_pred_test[:, i] = channel_preds

        self.logger.info(f"Trained {len(forecasters)} forecasters")

        res_df = self._compute_eval_metrics(test_df, target_cols, y_pred_test)
        self.logger.info(f"Average RMSE: {res_df['rmse'].mean():.4f}")

        model_data = self._build_model_artifact(
            forecasters=forecasters,
            target_cols=target_cols,
            exog_columns=exog_columns,
            channel_model_specs=channel_model_specs,
            diff_order=diff_order,
            n_lags=n_lags,
            model_label=model_label,
            train_ratio=train_ratio,
            train_df=train_df,
            test_df=test_df,
            timestamp=timestamp,
        )

        if save_model:
            self._save_model_artifact(model_data, panel_id, timestamp)

        return res_df, timestamp

    def _prepare_panel_data(
        self,
        panel_data: pd.DataFrame,
        weight_suffix: str,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Split columns into target/exog and apply known-anomaly masking to targets."""
        target_cols, exog_columns = _split_panel_columns(self.config, self.logger, panel_data, weight_suffix)

        known_anomalies = self.config.get("known_anomalies", [])
        known_anomaly_buffer = self.config["train"].get("known_anomaly_buffer")
        if known_anomalies and known_anomaly_buffer:
            panel_data = _mask_known_anomalies(panel_data, known_anomalies, known_anomaly_buffer, columns=target_cols)

        return panel_data, target_cols, exog_columns

    def _split_train_test(
        self,
        panel_data: pd.DataFrame,
        train_ratio: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Time-series split with frequency normalization on both halves."""
        train_df, test_df = _time_series_train_test_split(panel_data, train_ratio=train_ratio)
        fallback_freq = self.config.get("process", {}).get("resample", {}).get("freq", "5min")
        train_df = _ensure_freq(train_df, fallback_freq)
        test_df = _ensure_freq(test_df, fallback_freq)
        self.logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        return train_df, test_df

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
        n_lags: int | list[int],
    ) -> tuple[int | list[int], int]:
        """Return (effective_lags, effective_n_lags) from per-channel ``best_lags`` if any.

        ``effective_lags`` is what we set on the forecaster (may be a list of
        specific lag offsets). ``effective_n_lags`` is the int used for
        sufficiency checks (max lag = longest history needed).
        """
        effective = channel_cfg.get("best_lags", n_lags)
        if isinstance(effective, (list, tuple)) and effective:
            return list(effective), int(max(effective))
        try:
            n = int(effective)
            return n, n
        except (TypeError, ValueError):
            if isinstance(n_lags, (list, tuple)) and n_lags:
                return list(n_lags), int(max(n_lags))
            return n_lags, int(n_lags) if not isinstance(n_lags, (list, tuple)) else 24

    def _compute_training_weights(
        self,
        observed_mask: pd.Series,
        effective_n_lags: int,
        y_train: pd.Series,
        target_col: str,
    ) -> pd.Series | str | None:
        """Combine strict-imputation mask + optional auto-clean anomaly mask.

        Returns:
            - ``None`` if no weighting is needed
            - ``"skip"`` sentinel if the strict mask leaves zero usable samples
            - ``pd.Series`` of bool weights otherwise
        """
        sample_mask: pd.Series | None = None
        train_cfg = self.config.get("train", {})

        if train_cfg.get("exclude_imputed_training_samples", False):
            sample_mask = _build_strict_training_sample_mask(observed_mask=observed_mask, n_lags=effective_n_lags)
            potential = max(len(y_train) - effective_n_lags, 0)
            kept = int(sample_mask.iloc[effective_n_lags:].sum())
            self.logger.info(
                f"    {target_col}: excluding {potential - kept} training sample(s) "
                f"that contain imputed target/lag values"
            )
            if kept == 0:
                return "skip"

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
                if sample_mask is not None:
                    sample_mask = sample_mask & ~anomaly_mask
                else:
                    sample_mask = ~anomaly_mask

        return sample_mask

    def _fit_channel_forecaster(
        self,
        model_name: str,
        model_params: dict[str, Any],
        effective_lags: int | list[int],
        random_seed: int,
        y_train: pd.Series,
        exog_train: pd.DataFrame | None,
        sample_mask: pd.Series | None,
        diff_order: int,
    ):
        """Create forecaster, attach weight_func, fit (with Δy when needed), detach for serialization."""
        forecaster = _create_forecaster(
            model_name,
            model_params,
            n_lags=effective_lags,
            random_seed=random_seed,
            has_exog=exog_train is not None,
            logger=self.logger,
        )

        if sample_mask is not None:

            def _weight_func(index, mask=sample_mask):
                return mask.reindex(index).fillna(False).astype(float).to_numpy()

            forecaster.weight_func = _weight_func

        # Fit on Δy when differentiation > 0. spotforecast2's transformer_y /
        # weight_func still apply normally to the differenced target.
        if diff_order > 0:
            y_for_fit = _difference(y_train, diff_order).dropna()
            exog_for_fit = exog_train.loc[y_for_fit.index] if exog_train is not None else None
        else:
            y_for_fit = y_train
            exog_for_fit = exog_train
        forecaster.fit(y=y_for_fit, exog=exog_for_fit)

        # Detach the closure so joblib can serialise the fitted forecaster.
        if sample_mask is not None:
            forecaster.weight_func = None
            forecaster.source_code_weight_func = None

        return forecaster

    def _predict_test_window(
        self,
        forecaster,
        y_train: pd.Series,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        weight_suffix: str,
        exog_columns: list[str],
        diff_order: int,
    ) -> np.ndarray:
        """One-step-ahead test predictions on real observed lags.

        Mirrors what detection / live mode does at scoring time (see
        ``_predict_one_step_integrated``), so the train-time eval metric is
        the same one production reports.
        """
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
            return _predict_one_step_integrated(forecaster, full_y_raw, full_exog, test_df.index, diff_order)
        except Exception as exc:
            self.logger.warning(f"  One-step-ahead test prediction failed for {target_col}: {exc}")
            return np.full(len(test_df), np.nan)

    def _train_channel(
        self,
        target_col: str,
        channel_cfg: dict[str, Any],
        panel_default_model: str | None,
        panel_default_params: dict[str, Any],
        model_label: str,
        n_lags: int | list[int],
        random_seed: int,
        diff_order: int,
        weight_suffix: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        exog_columns: list[str],
    ) -> tuple[Any, dict[str, Any], np.ndarray] | None:
        """Train one forecaster + run test eval for a single channel.

        Returns ``(forecaster, model_spec, test_predictions)`` or ``None`` when
        the channel is skipped (insufficient data or all training samples
        masked out).
        """
        channel_model_name, channel_model_params = self._resolve_channel_model_spec(
            channel_cfg, panel_default_model, panel_default_params, model_label
        )
        effective_lags, effective_n_lags = self._resolve_channel_lags(channel_cfg, n_lags)

        self.logger.info(f"  Training forecaster for: {target_col} (model={channel_model_name})")

        y_train_raw = train_df[target_col].copy()
        y_train_raw.name = target_col

        observed_mask = _compute_observed_mask(train_df, target_col, weight_suffix)
        y_train = y_train_raw.copy()
        y_train.loc[~observed_mask] = np.nan
        if y_train.isna().any():
            y_train = _interpolate_inplace(y_train)
        if len(y_train) < effective_n_lags + 10:
            self.logger.warning(f"  Skipping {target_col}: insufficient data ({len(y_train)} rows)")
            return None

        sample_mask = self._compute_training_weights(observed_mask, effective_n_lags, y_train, target_col)
        if sample_mask == "skip":
            self.logger.warning(f"  Skipping {target_col}: no fully observed training samples left")
            return None

        exog_train = train_df[exog_columns].loc[y_train.index] if exog_columns else None
        forecaster = self._fit_channel_forecaster(
            channel_model_name,
            channel_model_params,
            effective_lags,
            random_seed,
            y_train,
            exog_train,
            sample_mask if isinstance(sample_mask, pd.Series) else None,
            diff_order,
        )

        channel_preds = self._predict_test_window(
            forecaster, y_train, train_df, test_df, target_col, weight_suffix, exog_columns, diff_order
        )

        y_test_col = test_df[target_col].values
        valid = ~np.isnan(y_test_col)
        if valid.any():
            rmse = np.sqrt(np.mean((y_test_col[valid] - channel_preds[valid]) ** 2))
            mae = np.mean(np.abs(y_test_col[valid] - channel_preds[valid]))
            self.logger.info(f"    {target_col}: RMSE={rmse:.4f}, MAE={mae:.4f}")

        return (
            forecaster,
            {
                "model": channel_model_name,
                "params": channel_model_params,
                "lags": effective_lags,
            },
            channel_preds,
        )

    @staticmethod
    def _compute_eval_metrics(
        test_df: pd.DataFrame,
        target_cols: list[str],
        y_pred_test: np.ndarray,
    ) -> pd.DataFrame:
        """Build per-channel RMSE/MAE DataFrame from the test predictions."""
        y_test_vals = test_df[target_cols].values
        rmse = np.sqrt(np.nanmean((y_test_vals - y_pred_test) ** 2, axis=0))
        mae = np.nanmean(np.abs(y_test_vals - y_pred_test), axis=0)
        return pd.DataFrame({"rmse": rmse, "mae": mae}, index=target_cols)

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
        train_ratio: float,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        timestamp: str,
    ) -> dict[str, Any]:
        """Package the dict that ``SpotforecastPredictor`` consumes."""
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
            "train_ratio": train_ratio,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_start_timestamp": self._iso_min(train_df),
            "train_end_timestamp": self._iso_max(train_df),
            "test_start_timestamp": self._iso_min(test_df),
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
