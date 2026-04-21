"""Adapter wrapping spotforecast2 MultiTask API for use in spotanomaly2's pipeline.

Uses ``MultiTask`` from ``spotforecast2.manager.multitask`` as the standardised
entry-point for forecaster creation, training, model persistence, and loading.
spotanomaly2's own data-processing pipeline (5-minute resampling, imputation, etc.)
runs *before* this adapter, so we bypass ``MultiTask.prepare_data()`` and drive
the per-channel training loop directly.
"""

from pathlib import Path
from typing import Any, Optional
import inspect

import joblib
import numpy as np
import pandas as pd
import yaml

from spotanomaly2.infrastructure import logging, storage
from spotanomaly2.infrastructure.storage import generate_timestamp


try:
    from catboost import CatBoostRegressor as _CatBoostBase

    class _CatBoostRegressor(_CatBoostBase):
        """Suppress set_params errors on fitted models.

        spotforecast2 calls set_params(task_type='CPU') during prediction
        on an already-fitted model; CatBoost rejects this via its C++ backend.
        Defined at module level so joblib/pickle can resolve the class path.
        """
        @property
        def task_type(self):
            return "CPU"

        def set_params(self, **params):
            try:
                return super().set_params(**params)
            except Exception:
                return self

except ImportError:
    _CatBoostRegressor = None  # type: ignore[assignment,misc]


def _time_series_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _mask_known_anomalies(
    df: pd.DataFrame,
    known_anomalies: list[dict],
    buffer: str,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Set selected columns to NaN within known anomaly windows (extended by buffer)."""
    if not known_anomalies:
        return df
    buffer_td = pd.Timedelta(buffer)
    df = df.copy()
    cols_to_mask = columns if columns is not None else list(df.columns)
    for anomaly in known_anomalies:
        start_str = anomaly.get("start")
        end_str = anomaly.get("end")
        if not start_str or not end_str:
            continue
        try:
            a_start = pd.to_datetime(start_str)
            a_end = pd.to_datetime(end_str)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                a_start = a_start.tz_convert(df.index.tz) if a_start.tz else a_start.tz_localize(df.index.tz)
                a_end = a_end.tz_convert(df.index.tz) if a_end.tz else a_end.tz_localize(df.index.tz)
            mask = (df.index >= a_start - buffer_td) & (df.index <= a_end + buffer_td)
            existing_cols = [c for c in cols_to_mask if c in df.columns]
            if existing_cols:
                df.loc[mask, existing_cols] = np.nan
        except (ValueError, TypeError):
            continue
    return df


class SpotforecastTrainer:
    """Wraps spotforecast2 MultiTask API to match spotanomaly2's interface.

    Uses ``MultiTask`` from ``spotforecast2.manager.multitask`` for standardised
    forecaster creation (with rolling-window features) while keeping
    spotanomaly2's own per-panel training/prediction loop for 5-minute data.
    """

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("SpotforecastTrainer")

    def _build_estimator(
        self,
        model_name: str,
        model_params: dict[str, Any],
        random_seed: int,
    ):
        """Build regressor instance for a channel-specific model configuration."""
        name = (model_name or "").strip().lower()
        params = dict(model_params or {})

        def _filter_params(estimator_cls, raw_params: dict[str, Any]) -> dict[str, Any]:
            sig = inspect.signature(estimator_cls.__init__)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if has_var_kw:
                return raw_params

            valid_keys = {key for key in sig.parameters if key != "self"}
            filtered = {k: v for k, v in raw_params.items() if k in valid_keys}
            dropped = sorted(k for k in raw_params if k not in valid_keys)
            if dropped:
                self.logger.warning(
                    f"Ignoring unsupported params for {model_name}: {dropped}"
                )
            return filtered

        if name in {"lightgbm", "lgbm", "lgbmregressor"}:
            from lightgbm import LGBMRegressor

            params.setdefault("random_state", random_seed)
            params = _filter_params(LGBMRegressor, params)
            return LGBMRegressor(**params)

        if name in {"xgboost", "xgb", "xgbregressor"}:
            from xgboost import XGBRegressor

            params.setdefault("random_state", random_seed)
            params = _filter_params(XGBRegressor, params)
            return XGBRegressor(**params)

        if name in {"ridge", "ridgeregressor"}:
            from sklearn.linear_model import Ridge

            params = _filter_params(Ridge, params)
            return Ridge(**params)

        if name in {"catboost", "catboostregressor"}:
            if _CatBoostRegressor is None:
                raise ImportError("catboost is not installed")
            params.setdefault("random_seed", random_seed)
            params.setdefault("verbose", 0)
            params = _filter_params(_CatBoostRegressor, params)
            return _CatBoostRegressor(**params)

        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: LightGBM, XGBoost, Ridge, CatBoost"
        )

    def _load_panel_channel_config(self, panel_id: str) -> dict[str, Any]:
        """Load panel-specific channel model config YAML, if configured."""
        train_cfg = self.config.get("train", {})
        file_map = train_cfg.get("channel_config_files", {})
        if not isinstance(file_map, dict):
            return {}

        cfg_path_value = file_map.get(panel_id) or file_map.get(f"panel_{panel_id}")
        if not cfg_path_value:
            return {}

        cfg_path = Path(cfg_path_value)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()

        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Channel model config for panel {panel_id} not found: {cfg_path}"
            )

        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

        if not isinstance(loaded, dict):
            raise ValueError(
                f"Channel model config for panel {panel_id} must be a mapping: {cfg_path}"
            )
        return loaded

    def _get_weight_suffix(self) -> str:
        return self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")

    def _observed_mask(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        """Return True for observed (non-imputed) rows of a target column."""
        weight_col = f"{target_col}{self._get_weight_suffix()}"
        if weight_col in df.columns:
            return df[weight_col].fillna(0.0) >= 0.5
        # If no weight column exists, treat rows as observed by default.
        return pd.Series(True, index=df.index)

    def _exclude_imputed_training_samples(self) -> bool:
        """Whether to zero-weight samples touching imputed values during fit."""
        return self.config.get("train", {}).get("exclude_imputed_training_samples", False)

    def _strict_training_sample_mask(
        self,
        observed_mask: pd.Series,
        n_lags: int,
    ) -> pd.Series:
        """Return True only when target and all required lag inputs are observed."""
        mask = observed_mask.fillna(False).astype(bool).copy()
        for lag in range(1, n_lags + 1):
            mask &= observed_mask.shift(lag).fillna(False).astype(bool)
        return mask

    def _detect_training_anomalies(
        self,
        y: pd.Series,
        n_lags: int,
        threshold_scale: float = 4.0,
        buffer: int = 3,
    ) -> pd.Series:
        """Detect anomalous regions in training data via a cheap Ridge pre-pass.

        Fits a Ridge AR model, computes one-step-ahead residuals, and flags
        points where |residual| exceeds ``threshold_scale * MAD`` (median
        absolute deviation).  Flagged regions are expanded by ``buffer`` steps
        on each side to catch the shoulders of level-shift anomalies.

        Returns a boolean Series (True = suspected anomaly) aligned to ``y``.
        """
        from sklearn.linear_model import Ridge

        clean_mask = pd.Series(False, index=y.index)
        n = len(y)
        lag = min(n_lags, 24)
        if n < lag + 20:
            return clean_mask

        # Build lag matrix from observed values
        X_rows = []
        y_rows = []
        indices = []
        vals = y.to_numpy(dtype=float)
        for t in range(lag, n):
            if np.any(np.isnan(vals[t - lag:t + 1])):
                continue
            X_rows.append(vals[t - lag:t][::-1])
            y_rows.append(vals[t])
            indices.append(y.index[t])

        if len(X_rows) < lag + 10:
            return clean_mask

        X = np.array(X_rows)
        y_arr = np.array(y_rows)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y_arr)
        preds = ridge.predict(X)
        resid = np.abs(y_arr - preds)

        # Robust threshold: median + k * MAD
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) or 1e-12
        threshold = med + threshold_scale * mad

        flagged = resid > threshold
        flagged_indices = {indices[i] for i, f in enumerate(flagged) if f}

        # Expand flagged regions by buffer on each side
        all_idx = list(y.index)
        expanded: set = set()
        for idx in flagged_indices:
            try:
                pos = all_idx.index(idx)
            except ValueError:
                continue
            for offset in range(-buffer, buffer + 1):
                p = pos + offset
                if 0 <= p < n:
                    expanded.add(all_idx[p])

        clean_mask.loc[list(expanded)] = True
        return clean_mask

    def _create_multitask(self, panel_id: str, n_lags: int, cache_home: Path | None = None):
        """Create a configured ``MultiTask`` instance for forecaster creation.

        The returned object is *not* run through ``prepare_data`` because
        spotanomaly2 already provides pre-processed 5-minute data.  We only
        use it for ``create_forecaster()`` which builds a properly configured
        ``ForecasterRecursive`` with rolling-window features.
        """
        from spotforecast2.manager.multitask import MultiTask

        random_seed = self.config["train"].get("random_seed", 42)

        return MultiTask(
            task="lazy",
            data_frame_name=f"panel_{panel_id}",
            predict_size=24,
            cache_home=cache_home,
            use_exogenous_features=False,
            use_outlier_detection=False,
            lags_consider=list(range(1, n_lags + 1)),
            random_state=random_seed,
            auto_save_models=False,
        )

    def train_panel(
        self,
        panel_id: str,
        df: pd.DataFrame,
        timestamp: Optional[str] = None,
        save_model: bool = True,
        panel_specific_params: Optional[dict[str, Any]] = None,
        channel_specific_params: Optional[dict[str, Any]] = None,
    ) -> tuple[pd.DataFrame, str]:
        """Train spotforecast2 models for a panel via MultiTask.

        Creates one ``ForecasterRecursive`` per target channel using
        ``MultiTask.create_forecaster()`` for standardised model configuration,
        then fits each forecaster on the training split.
        """
        base_models_dir = Path(self.config["paths"]["models_dir"])
        train_ratio = self.config["train"]["train_ratio"]
        model_label = self.config["train"]["model"]
        n_lags = self.config["train"].get("lags", 24)
        random_seed = self.config["train"].get("random_seed", 42)

        if timestamp is None:
            timestamp = generate_timestamp()
        models_dir = base_models_dir / timestamp

        self.logger.info(
            f"Training spotforecast2 models on {len(df)} rows for panel {panel_id}"
        )

        configured_exog_columns = self.config["train"].get("exog_columns", [])

        # When weight_residuals is enabled, exogenous columns are used only
        # for post-prediction residual weighting — not as model features.
        weight_residuals_enabled = (
            self.config.get("exogenous", {}).get("weight_residuals", {}).get("enabled", False)
        )
        if weight_residuals_enabled:
            prefixed_exog_columns: list[str] = []
            self.logger.info(
                "weight_residuals enabled: exogenous columns excluded from model features"
            )
        else:
            prefixed_exog_columns = [
                col for col in df.columns
                if col.startswith("exogenous_")
            ]
        exog_columns = [
            col
            for col in [*configured_exog_columns, *prefixed_exog_columns]
            if col in df.columns
        ]
        # Keep order stable while removing duplicates.
        exog_columns = list(dict.fromkeys(exog_columns))
        weight_suffix = self._get_weight_suffix()
        target_cols = [
            col for col in df.columns
            if not col.endswith(weight_suffix)
            and col not in exog_columns
            and not col.startswith("exogenous_")
        ]

        known_anomalies = self.config.get("known_anomalies", [])
        known_anomaly_buffer = self.config["train"].get("known_anomaly_buffer")
        if known_anomalies and known_anomaly_buffer:
            df = _mask_known_anomalies(
                df,
                known_anomalies,
                known_anomaly_buffer,
                columns=target_cols,
            )

        train_df, test_df = _time_series_train_test_split(df, train_ratio=train_ratio)

        if isinstance(train_df.index, pd.DatetimeIndex) and train_df.index.freq is None:
            inferred_freq = pd.infer_freq(train_df.index)
            fallback_freq = self.config.get("process", {}).get("resample", {}).get("freq", "5min")
            target_freq = inferred_freq or fallback_freq
            train_df = train_df.asfreq(target_freq)
            test_df = test_df.asfreq(target_freq)
            self.logger.info(f"Using training frequency: {target_freq}")

        self.logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        self.logger.info("Training per-channel ForecasterRecursive models via MultiTask...")

        mt = self._create_multitask(panel_id, n_lags, cache_home=models_dir)

        panel_cfg_from_yaml = self._load_panel_channel_config(panel_id)
        if panel_specific_params:
            merged_panel_cfg = dict(panel_cfg_from_yaml)
            merged_panel_cfg.update(panel_specific_params)
            panel_cfg_from_yaml = merged_panel_cfg

        panel_default_model = (
            panel_cfg_from_yaml.get("default", {}).get("model")
            if isinstance(panel_cfg_from_yaml.get("default"), dict)
            else panel_cfg_from_yaml.get("default_model")
        )
        panel_default_params = (
            panel_cfg_from_yaml.get("default", {}).get("params", {})
            if isinstance(panel_cfg_from_yaml.get("default"), dict)
            else panel_cfg_from_yaml.get("default_params", {})
        )
        channel_cfg_map = panel_cfg_from_yaml.get("channels", {})
        if channel_specific_params:
            merged_channel_cfg = dict(channel_cfg_map)
            merged_channel_cfg.update(channel_specific_params)
            channel_cfg_map = merged_channel_cfg

        forecasters: dict[str, Any] = {}
        channel_model_specs: dict[str, dict[str, Any]] = {}
        y_pred_test = np.zeros((len(test_df), len(target_cols)))

        for i, target_col in enumerate(target_cols):
            self.logger.info(f"  Training forecaster for: {target_col}")

            # Resolve per-channel lag spec from the panel YAML (best_lags).
            # Falls back to the global train.lags. ``effective_lags`` is what
            # we'll set on the forecaster; ``effective_n_lags`` is the int
            # used for sufficiency checks and the strict-observed mask.
            channel_cfg_for_lags = channel_cfg_map.get(target_col, {})
            if not isinstance(channel_cfg_for_lags, dict):
                channel_cfg_for_lags = {}
            effective_lags: Any = channel_cfg_for_lags.get("best_lags", n_lags)
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

            observed_mask = self._observed_mask(train_df, target_col)
            y_train = y_train_raw.copy()
            y_train.loc[~observed_mask] = np.nan
            if y_train.isna().any():
                if isinstance(y_train.index, pd.DatetimeIndex):
                    y_train = y_train.interpolate(method="time").bfill().ffill()
                else:
                    y_train = y_train.interpolate(method="linear").bfill().ffill()
            if len(y_train) < effective_n_lags + 10:
                self.logger.warning(f"  Skipping {target_col}: insufficient data ({len(y_train)} rows)")
                continue

            strict_observed_training = self._exclude_imputed_training_samples()
            sample_mask_for_weights = None
            if strict_observed_training:
                sample_mask_for_weights = self._strict_training_sample_mask(
                    observed_mask=observed_mask,
                    n_lags=effective_n_lags,
                )
                potential_rows = max(len(y_train) - effective_n_lags, 0)
                kept_rows = int(sample_mask_for_weights.iloc[effective_n_lags:].sum())
                dropped_rows = potential_rows - kept_rows
                self.logger.info(
                    f"    {target_col}: excluding {dropped_rows} training sample(s) "
                    f"that contain imputed target/lag values"
                )
                if kept_rows == 0:
                    self.logger.warning(f"  Skipping {target_col}: no fully observed training samples left")
                    continue

            # Two-pass anomaly cleaning: detect anomalous training regions
            # with a cheap Ridge pre-pass and zero-weight them alongside
            # imputed samples.
            auto_clean = self.config.get("train", {}).get("auto_clean_anomalies", False)
            if auto_clean:
                anomaly_mask = self._detect_training_anomalies(
                    y_train, n_lags=effective_n_lags,
                    threshold_scale=self.config.get("train", {}).get(
                        "auto_clean_threshold", 4.0),
                    buffer=self.config.get("train", {}).get(
                        "auto_clean_buffer", 3),
                )
                n_flagged = int(anomaly_mask.sum())
                if n_flagged > 0:
                    self.logger.info(
                        f"    {target_col}: auto-cleaning flagged {n_flagged} "
                        f"suspected anomaly points in training data"
                    )
                    if sample_mask_for_weights is not None:
                        sample_mask_for_weights = sample_mask_for_weights & ~anomaly_mask
                    else:
                        sample_mask_for_weights = ~anomaly_mask

            exog_train = None
            exog_test = None
            if exog_columns:
                exog_train = train_df[exog_columns].loc[y_train.index]
                exog_test = test_df[exog_columns]

            forecaster = mt.create_forecaster()

            # Apply per-channel lag spec resolved above so tuned best_lags
            # actually drive the live forecaster (previously ignored —
            # train.lags was the only knob that took effect).
            if effective_lags != n_lags:
                try:
                    forecaster.set_lags(effective_lags)
                except Exception as exc:
                    self.logger.warning(
                        f"  {target_col}: failed to apply best_lags={effective_lags} ({exc}); "
                        f"falling back to global lags={n_lags}"
                    )

            channel_cfg = channel_cfg_for_lags

            channel_model_name = (
                channel_cfg.get("model")
                or panel_default_model
                or model_label
            )
            default_name_for_params = (
                (panel_default_model or model_label or "").strip().lower()
            )
            channel_name_for_params = (channel_model_name or "").strip().lower()
            if channel_name_for_params == default_name_for_params:
                channel_model_params = dict(panel_default_params or {})
            else:
                channel_model_params = {}
            channel_model_params.update(channel_cfg.get("params", {}) or {})

            forecaster.estimator = self._build_estimator(
                channel_model_name,
                channel_model_params,
                random_seed=random_seed,
            )

            if sample_mask_for_weights is not None:

                def _weight_func(index, mask=sample_mask_for_weights):
                    return mask.reindex(index).fillna(False).astype(float).to_numpy()

                forecaster.weight_func = _weight_func

            forecaster.fit(y=y_train, exog=exog_train)

            # Weight callback is only needed during fit; keeping a nested
            # function attached breaks model pickling in joblib.
            if sample_mask_for_weights is not None:
                forecaster.weight_func = None
                forecaster.source_code_weight_func = None

            channel_model_specs[target_col] = {
                "model": channel_model_name,
                "params": channel_model_params,
                "lags": effective_lags,
            }

            steps = len(test_df)
            exog_for_pred = None
            if exog_test is not None:
                exog_for_pred = exog_test.copy()
                if isinstance(y_train.index, pd.DatetimeIndex) and y_train.index.freq is not None:
                    pred_index = pd.date_range(
                        start=y_train.index[-1] + y_train.index.freq,
                        periods=steps,
                        freq=y_train.index.freq,
                    )
                    exog_for_pred.index = pred_index

            preds = forecaster.predict(steps=steps, exog=exog_for_pred)
            if len(preds) == len(test_df):
                y_pred_test[:, i] = preds.values
            else:
                aligned = preds.reindex(test_df.index)
                y_pred_test[:, i] = aligned.values

            forecasters[target_col] = forecaster

            y_test_col = test_df[target_col].values
            valid = ~np.isnan(y_test_col)
            if valid.any():
                rmse = np.sqrt(np.mean((y_test_col[valid] - y_pred_test[valid, i]) ** 2))
                mae = np.mean(np.abs(y_test_col[valid] - y_pred_test[valid, i]))
                self.logger.info(f"    {target_col}: RMSE={rmse:.4f}, MAE={mae:.4f}")

        self.logger.info(f"Trained {len(forecasters)} forecasters via MultiTask")

        y_test_vals = test_df[target_cols].values
        rmse_results = np.sqrt(np.nanmean((y_test_vals - y_pred_test) ** 2, axis=0))
        mae_results = np.nanmean(np.abs(y_test_vals - y_pred_test), axis=0)

        res_df = pd.DataFrame(
            {
                "rmse": rmse_results,
                "mae": mae_results,
            },
            index=target_cols,
        )

        model_type_value = f"spotforecast2_{model_label.lower()}"
        model_data = {
            "forecasters": forecasters,
            "n_lags": n_lags,
            "target_cols": target_cols,
            "exog_columns": exog_columns,
            "channel_models": channel_model_specs,
            "model_type": model_type_value,
            "model_name": model_label,
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
            model_filename = f"{model_label}_fc_model_panel_{panel_id}.pkl"
            model_path = models_dir / model_filename
            joblib.dump(model_data, model_path)
            self.logger.info(f"Saved spotforecast2 model to {model_path}")

        self.logger.info(f"Average RMSE: {rmse_results.mean():.4f}")
        return res_df, timestamp

    def predict(
        self,
        model_data: dict[str, Any],
        df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate one-step-ahead predictions using trained ForecasterRecursive models.

        Uses ``create_train_X_y`` to build lag/rolling-window features from
        the *actual* observed values at every timestep, then runs the
        underlying estimator in a single batch.  This gives true
        one-step-ahead forecasts suitable for anomaly detection (each
        prediction is conditioned on real data, not on prior predictions).
        """
        forecasters = model_data["forecasters"]
        target_cols = model_data["target_cols"]
        exog_columns = model_data.get("exog_columns", [])

        def _ensure_freq(frame: pd.DataFrame) -> pd.DataFrame:
            if isinstance(frame.index, pd.DatetimeIndex) and frame.index.freq is None:
                inferred = pd.infer_freq(frame.index)
                if inferred:
                    frame = frame.asfreq(inferred)
            return frame

        df = _ensure_freq(df)
        if history_df is not None:
            history_df = _ensure_freq(history_df)

        predictions = {}

        for target_col in target_cols:
            if target_col not in forecasters:
                predictions[target_col] = np.full(len(df), np.nan)
                continue

            forecaster = forecasters[target_col]

            if history_df is not None:
                full_y = pd.concat([history_df[target_col], df[target_col]])
                observed_mask = pd.concat(
                    [
                        self._observed_mask(history_df, target_col),
                        self._observed_mask(df, target_col),
                    ]
                )
            else:
                full_y = df[target_col].copy()
                observed_mask = self._observed_mask(df, target_col)
            full_y.name = target_col

            full_y = full_y.copy()
            full_y.loc[~observed_mask] = np.nan
            if full_y.isna().any():
                if isinstance(full_y.index, pd.DatetimeIndex):
                    full_y = full_y.interpolate(method="time").bfill().ffill()
                else:
                    full_y = full_y.interpolate(method="linear").bfill().ffill()
            if len(full_y) == 0:
                predictions[target_col] = np.full(len(df), np.nan)
                continue

            if isinstance(full_y.index, pd.DatetimeIndex) and full_y.index.freq is None:
                inferred = pd.infer_freq(full_y.index)
                if inferred:
                    full_y.index = pd.DatetimeIndex(full_y.index, freq=inferred)

            exog_full = None
            if exog_columns:
                exog_cols_present = [c for c in exog_columns if c in df.columns]
                if exog_cols_present:
                    if history_df is not None:
                        exog_full = pd.concat(
                            [
                                history_df[exog_cols_present],
                                df[exog_cols_present],
                            ]
                        )
                    else:
                        exog_full = df[exog_cols_present].copy()
                    exog_full = exog_full.loc[full_y.index]

            try:
                x_features, y_target = forecaster.create_train_X_y(
                    y=full_y,
                    exog=exog_full,
                )
                preds_all = forecaster.estimator.predict(x_features)

                # Align to requested df index
                preds_series = pd.Series(preds_all, index=y_target.index, name=target_col)
                aligned = preds_series.reindex(df.index)
                predictions[target_col] = aligned.values
            except Exception as e:
                self.logger.warning(f"Prediction failed for {target_col}: {e}. Using NaN.")
                predictions[target_col] = np.full(len(df), np.nan)

        pred_df = pd.DataFrame(predictions, index=df.index)
        return pred_df

    def train_all_panels(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        training_timestamp = generate_timestamp()
        results = {}
        for panel_id, df in panel_data.items():
            eval_df, ts = self.train_panel(panel_id, df, timestamp=training_timestamp)
            results[panel_id] = (eval_df, ts)
        return results

    def run(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        self.logger.info("Starting spotforecast2 model training...")
        results = self.train_all_panels(panel_data)
        self.logger.info("spotforecast2 model training completed")
        return results


def _yaml_search_space_to_dict(raw: dict) -> dict:
    """Convert YAML search space to the dict format expected by spotoptim_search_forecaster.

    YAML lists of 2 elements become tuples (low, high), lists of 3 elements become
    (low, high, transform), and plain lists stay as categorical factors.
    """
    result = {}
    for key, value in raw.items():
        if isinstance(value, list):
            if len(value) in (2, 3) and all(isinstance(v, (int, float)) for v in value[:2]):
                result[key] = tuple(value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result



class SpotforecastTuner:
    """Tunes spotforecast2 ForecasterRecursive hyperparameters via SpotOptim.

    Uses ``spotoptim_search_forecaster`` from ``spotforecast2.model_selection``
    to run surrogate-model Bayesian optimisation per target channel.
    Supports LightGBM, XGBoost, and CatBoost estimators.
    """

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("SpotforecastTuner")

    def _get_weight_suffix(self) -> str:
        return self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")

    def _create_forecaster(
        self,
        model_name: str,
        n_lags: int,
        random_seed: int,
        window_sizes: list[int] | None = None,
    ):
        """Create a ForecasterRecursive with the specified estimator."""
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2_safe.preprocessing import RollingFeatures as RollingFeaturesUnified

        # Reuse the trainer's _build_estimator so all model names are supported.
        trainer = SpotforecastTrainer(self.config, self.logger)
        estimator = trainer._build_estimator(model_name, {}, random_seed)
        kwargs: dict[str, Any] = {"estimator": estimator, "lags": n_lags}
        if window_sizes:
            kwargs["window_features"] = RollingFeaturesUnified(
                stats=["mean"],
                window_sizes=window_sizes,
            )
        return ForecasterRecursive(**kwargs)

    def tune_panel(
        self,
        panel_id: str,
        df: pd.DataFrame,
        tune_config: dict[str, Any],
        channels: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Tune hyperparameters for each channel in a panel via SpotOptim.

        Args:
            panel_id: Panel identifier.
            df: Processed panel DataFrame.
            tune_config: Merged tuning configuration.
            channels: If provided, only tune these channel column names.

        Returns:
            Dict mapping channel_name -> {best_params, best_lags, best_metric, best_model}.
        """
        from spotforecast2.model_selection import spotoptim_search_forecaster
        from spotforecast2.model_selection.split_ts_cv import TimeSeriesFold
        from tqdm.auto import tqdm

        train_ratio = self.config["train"]["train_ratio"]
        n_lags = self.config["train"].get("lags", 24)
        random_seed = self.config["train"].get("random_seed", 42)

        known_anomalies = self.config.get("known_anomalies", [])
        known_anomaly_buffer = self.config["train"].get("known_anomaly_buffer")
        if known_anomalies and known_anomaly_buffer:
            df = _mask_known_anomalies(df, known_anomalies, known_anomaly_buffer)

        exog_columns = [col for col in self.config["train"].get("exog_columns", []) if col in df.columns]
        weight_suffix = self._get_weight_suffix()
        target_cols = [col for col in df.columns if not col.endswith(weight_suffix) and col not in exog_columns]

        if channels is not None:
            target_cols = [c for c in target_cols if c in channels]
            missing = set(channels) - set(target_cols)
            if missing:
                self.logger.warning(f"Requested channels not found in data: {missing}")

        train_df, _ = _time_series_train_test_split(df, train_ratio=train_ratio)

        if isinstance(train_df.index, pd.DatetimeIndex) and train_df.index.freq is None:
            inferred_freq = pd.infer_freq(train_df.index)
            fallback_freq = self.config.get("process", {}).get("resample", {}).get("freq", "5min")
            target_freq = inferred_freq or fallback_freq
            train_df = train_df.asfreq(target_freq)

        n_trials = tune_config.get("n_trials", 10)
        n_initial = tune_config.get("n_initial", 5)
        metric = tune_config.get("metric", "mean_absolute_error")
        model_search_spaces = tune_config.get("model_search_spaces", {})

        models = tune_config.get("models", ["LightGBM"])
        if isinstance(models, str):
            models = [models]
        # Validate model names via _build_estimator
        trainer = SpotforecastTrainer(self.config, self.logger)
        valid_models = []
        for m in models:
            try:
                trainer._build_estimator(m, {}, 0)
                valid_models.append(m)
            except (ValueError, ImportError):
                self.logger.warning(f"Model '{m}' not available, skipping")
        models = valid_models or ["LightGBM"]

        self.logger.info(
            f"Tuning panel {panel_id}: {len(target_cols)} channels, "
            f"{n_trials} trials, {n_initial} initial, models={models}"
        )

        steps = max(1, int(len(train_df) * 0.1))
        cv = TimeSeriesFold(steps=steps, initial_train_size=int(len(train_df) * 0.7))

        results: dict[str, dict[str, Any]] = {}

        channel_pbar = tqdm(
            target_cols,
            desc=f"Panel {panel_id}",
            unit="channel",
            leave=True,
        )
        for target_col in channel_pbar:
            channel_pbar.set_postfix_str(target_col)

            channel_n_trials = n_trials
            channel_n_initial = n_initial

            panel_overrides = tune_config.get("panel_overrides", {}).get(panel_id, {})
            if panel_overrides.get("default"):
                channel_n_trials = panel_overrides["default"].get("n_trials", channel_n_trials)
                channel_n_initial = panel_overrides["default"].get("n_initial", channel_n_initial)
            channel_overrides = panel_overrides.get("channels", {}).get(target_col, {})
            if channel_overrides:
                channel_n_trials = channel_overrides.get("n_trials", channel_n_trials)
                channel_n_initial = channel_overrides.get("n_initial", channel_n_initial)

            y_train = train_df[target_col].copy()
            y_train.name = target_col
            if y_train.isna().any():
                if isinstance(y_train.index, pd.DatetimeIndex):
                    y_train = y_train.interpolate(method="time").bfill().ffill()
                else:
                    y_train = y_train.interpolate(method="linear").bfill().ffill()

            if len(y_train) < n_lags + 10:
                self.logger.warning(f"  Skipping {target_col}: insufficient data ({len(y_train)} rows)")
                continue

            exog_train = None
            if exog_columns:
                exog_train = train_df[exog_columns].loc[y_train.index]
                if exog_train.isna().any().any():
                    if isinstance(exog_train.index, pd.DatetimeIndex):
                        exog_train = exog_train.interpolate(method="time").bfill().ffill()
                    else:
                        exog_train = exog_train.interpolate(method="linear").bfill().ffill()

            best_overall: dict[str, Any] | None = None

            model_pbar = tqdm(
                models,
                desc=f"  {target_col}",
                unit="model",
                leave=False,
            )
            for model_name in model_pbar:
                model_pbar.set_postfix_str(model_name)

                try:
                    forecaster = self._create_forecaster(model_name, n_lags, random_seed)
                    search_space_raw = model_search_spaces.get(model_name, {})
                    search_space = _yaml_search_space_to_dict(search_space_raw)

                    tuning_results, optimizer = spotoptim_search_forecaster(
                        forecaster=forecaster,
                        y=y_train,
                        cv=cv,
                        search_space=search_space,
                        metric=metric,
                        exog=exog_train,
                        return_best=True,
                        random_state=random_seed,
                        verbose=False,
                        n_trials=channel_n_trials,
                        n_initial=channel_n_initial,
                        show_progress=False,
                    )

                    best_row = tuning_results.iloc[0]
                    best_params = best_row["params"] if "params" in tuning_results.columns else {}
                    best_lags = best_row["lags"] if "lags" in tuning_results.columns else n_lags

                    non_meta_cols = [
                        c for c in tuning_results.columns if c not in ("params", "lags") and c not in best_params
                    ]
                    best_metric_val = None
                    if non_meta_cols:
                        raw_val = best_row[non_meta_cols[0]]
                        if hasattr(raw_val, "__len__") and not isinstance(raw_val, str):
                            best_metric_val = float(raw_val[0])
                        else:
                            best_metric_val = float(raw_val)

                    self.logger.info(
                        f"    {target_col} [{model_name}]: {metric}={best_metric_val:.6f}, lags={best_lags}"
                    )

                    if best_overall is None or (
                        best_metric_val is not None and best_metric_val < best_overall["best_metric"]
                    ):
                        best_overall = {
                            "best_params": best_params,
                            "best_lags": best_lags,
                            "best_metric": best_metric_val,
                            "best_model": model_name,
                        }

                except Exception as e:
                    self.logger.error(
                        f"  Tuning failed for {target_col} [{model_name}]: {e}",
                        exc_info=True,
                    )

            model_pbar.close()

            if best_overall is not None:
                results[target_col] = best_overall
                self.logger.info(
                    f"  >> {target_col}: winner={best_overall['best_model']}, "
                    f"{metric}={best_overall['best_metric']:.6f}"
                )
            else:
                results[target_col] = {"error": "All models failed"}

        channel_pbar.close()
        return results
