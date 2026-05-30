"""Per-panel hyperparameter tuning entry point: ``SpotforecastTuner``.

Wraps ``spotforecast2.model_selection.spotoptim_search_forecaster`` to run
surrogate-model Bayesian optimisation per target channel against every
candidate model in ``tune_config['models']``, picking the per-channel winner.

Mirrors ``SpotforecastTrainer.train_panel``'s preprocessing exactly (target/exog
split, imputation-flag handling, optional Ridge anomaly pre-pass, optional Δy
differentiation) so the tuned hyperparameters are optimal for the same objective
the trainer will actually fit. R² under differentiation is reported in raw-y
space via ``_build_raw_r2_under_differentiation`` so the tuner leaderboard
matches what production scoring shows.
"""

from typing import Any

import numpy as np
import pandas as pd

from spotanomaly2.domain.exogenous.residual_multiplier import multiplier_prefixes
from spotanomaly2.infrastructure import logging

from .factory import _build_estimator, _create_forecaster
from .prediction import _difference
from .preprocessing import (
    _build_strict_training_sample_mask,
    _compute_observed_mask,
    _detect_anomalies_via_ridge,
    _ensure_freq,
    _interpolate_inplace,
    _mask_known_anomalies,
    _split_panel_columns,
)
from .tuning_metrics import (
    _build_nan_safe_metric,
    _build_raw_r2_under_differentiation,
    _yaml_search_space_to_dict,
)


class SpotforecastTuner:
    """Tunes ForecasterRecursive hyperparameters per (channel, model) via SpotOptim.

    Uses ``spotoptim_search_forecaster`` from ``spotforecast2.model_selection``
    to run surrogate-model Bayesian optimisation per target channel for every
    candidate model in ``tune_config['models']``, picking the per-channel winner.
    """

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("SpotforecastTuner")

    def _get_weight_suffix(self) -> str:
        return self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")

    def tune_panel(
        self,
        panel_id: str,
        df: pd.DataFrame,
        tune_config: dict[str, Any],
        channels: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Tune hyperparameters for each channel in a panel via SpotOptim.

        Returns
        -------
        Dict mapping channel_name -> ``{best_params, best_lags, best_metric,
        best_model, model_scores}``. ``model_scores`` is a sub-dict per
        candidate model so callers can compare them, not just the winner.
        """
        from spotforecast2.model_selection import OneStepAheadFold, spotoptim_search_forecaster
        from tqdm.auto import tqdm

        n_lags = self.config["train"].get("lags", 24)
        random_seed = self.config["train"].get("random_seed", 42)
        exclude_imputed_training_samples = self.config.get("train", {}).get("exclude_imputed_training_samples", False)
        # Tune on Δy too — must match what train_panel fits, otherwise the
        # winning hyperparameters won't be optimal for the actual estimator
        # that ends up in production.
        diff_order = int(self.config.get("train", {}).get("differentiation", 1))
        # Conservative cap for the strict-sample mask: must cover the longest
        # lag the search may pick, so use the global config value as an upper
        # bound (search-space lag lists override per trial).
        if isinstance(n_lags, (list, tuple)) and n_lags:
            n_lags_max = int(max(n_lags))
        else:
            try:
                n_lags_max = int(n_lags)
            except (TypeError, ValueError):
                n_lags_max = 24

        known_anomalies = self.config.get("known_anomalies", [])
        known_anomaly_buffer = self.config["train"].get("known_anomaly_buffer")
        if known_anomalies and known_anomaly_buffer:
            df = _mask_known_anomalies(df, known_anomalies, known_anomaly_buffer)

        # Match SpotforecastTrainer.train_panel exactly: include `exogenous_*`
        # columns as exog features and exclude them from the tuning targets.
        # Otherwise tune optimizes a different model than train ends up fitting.
        configured_exog_columns = self.config["train"].get("exog_columns", [])
        weight_suffix = self._get_weight_suffix()
        mult_prefixes = multiplier_prefixes(self.config)
        target_cols, exog_columns = _split_panel_columns(df, configured_exog_columns, weight_suffix, mult_prefixes)

        if channels is not None:
            target_cols = [c for c in target_cols if c in channels]
            missing = set(channels) - set(target_cols)
            if missing:
                self.logger.warning(f"Requested channels not found in data: {missing}")

        # Use ALL the processed data — the trailing 10–20% carries the
        # strongest distribution drift; chopping it off would make CV blind
        # to live-shift. OneStepAheadFold below provides the train/val split.
        tune_df = _ensure_freq(df, self.config.get("process", {}).get("resample", {}).get("freq", "5min"))

        n_trials = tune_config.get("n_trials", 10)
        n_initial = tune_config.get("n_initial", 5)
        metric = tune_config.get("metric", "mean_absolute_error")
        # Wrap the string metric so a single divergent MLP / Bayesian / Huber
        # trial doesn't crash SpotOptim's surrogate fit (GaussianProcessRegressor
        # rejects NaN/Inf). Falls through unchanged when metric is already a
        # callable or an unknown string.
        metric_callable = _build_nan_safe_metric(metric) if isinstance(metric, str) else metric
        model_search_spaces = tune_config.get("model_search_spaces", {})

        models = tune_config.get("models", ["LightGBM"])
        if isinstance(models, str):
            models = [models]
        # Validate names via _build_estimator. Log the exact cause so a typo
        # or a missing dependency doesn't silently collapse the run to trees.
        valid_models: list[str] = []
        for m in models:
            try:
                _build_estimator(m, {}, random_seed=0, logger=self.logger)
                valid_models.append(m)
            except ValueError as exc:
                self.logger.warning(f"Skipping model '{m}': {exc}")
            except ImportError as exc:
                self.logger.warning(f"Skipping model '{m}': dependency missing ({exc})")
        models = valid_models or ["LightGBM"]

        self.logger.info(
            f"Tuning panel {panel_id}: {len(target_cols)} channels, "
            f"{n_trials} trials, {n_initial} initial, models={models}"
        )

        # OneStepAheadFold matches what production actually does at detection
        # time (see SpotforecastTrainer.predict). Train on the first 80%,
        # validate on the most recent 20% — that val window now spans the
        # part of the timeline closest to live conditions, so any
        # distribution drift inside the dataset shows up in the metric and
        # constant-mean predictors get exposed instead of tying real models.
        cv = OneStepAheadFold(initial_train_size=max(1, int(len(tune_df) * 0.8)), verbose=False)

        results: dict[str, dict[str, Any]] = {}

        channel_pbar = tqdm(target_cols, desc=f"Panel {panel_id}", unit="channel", leave=True)
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

            # Match train: respect imputation `__weight` flags so imputed rows
            # don't masquerade as observations during the search.
            observed_mask = _compute_observed_mask(tune_df, target_col, weight_suffix)
            y_train_raw = tune_df[target_col].copy()
            y_train_raw.name = target_col
            y_train = y_train_raw.copy()
            y_train.loc[~observed_mask] = np.nan
            if y_train.isna().any():
                y_train = _interpolate_inplace(y_train)
            if len(y_train) < n_lags_max + 10:
                self.logger.warning(f"  Skipping {target_col}: insufficient data ({len(y_train)} rows)")
                continue

            sample_mask_for_weights: pd.Series | None = None
            if exclude_imputed_training_samples:
                sample_mask_for_weights = _build_strict_training_sample_mask(
                    observed_mask=observed_mask,
                    n_lags=n_lags_max,
                )
                if int(sample_mask_for_weights.iloc[n_lags_max:].sum()) == 0:
                    self.logger.warning(f"  Skipping {target_col}: no fully observed training samples left")
                    continue

            train_cfg_section = self.config.get("train", {})
            if train_cfg_section.get("auto_clean_anomalies", False):
                anomaly_mask = _detect_anomalies_via_ridge(
                    y_train,
                    n_lags=n_lags_max,
                    threshold_scale=train_cfg_section.get("auto_clean_threshold", 4.0),
                    buffer=train_cfg_section.get("auto_clean_buffer", 3),
                )
                if int(anomaly_mask.sum()) > 0:
                    if sample_mask_for_weights is not None:
                        sample_mask_for_weights = sample_mask_for_weights & ~anomaly_mask
                    else:
                        sample_mask_for_weights = ~anomaly_mask

            exog_train = None
            if exog_columns:
                exog_train = tune_df[exog_columns].loc[y_train.index]
                if exog_train.isna().any().any():
                    exog_train = _interpolate_inplace(exog_train)

            # Predict Δy[t] just like train_panel will. The tuner must score
            # candidates on the same target the production model will fit
            # against, otherwise the winning hyperparameters are optimal
            # for a different objective.
            channel_metric_callable = metric_callable
            if diff_order > 0:
                # Capture the raw (pre-difference) series — the R² metric
                # looks up actual raw values by timestamp inside the val
                # window so the leaderboard number matches production.
                y_train_raw_pre_diff = y_train.copy()
                y_train = _difference(y_train, diff_order).dropna()
                if exog_train is not None:
                    exog_train = exog_train.loc[y_train.index]
                if sample_mask_for_weights is not None:
                    sample_mask_for_weights = sample_mask_for_weights.reindex(y_train.index).fillna(False)
                # Only R² needs the raw-space rescaling — MAE/MSE/RMSE
                # numerators are the same residuals in either space, so
                # they already match production without modification.
                if isinstance(metric, str) and metric.lower() in ("r2", "r2_score") and diff_order == 1:
                    channel_metric_callable = _build_raw_r2_under_differentiation(y_train_raw_pre_diff)

            best_overall: dict[str, Any] | None = None
            per_model: dict[str, dict[str, Any]] = {}

            model_pbar = tqdm(models, desc=f"  {target_col}", unit="model", leave=False)
            for model_name in model_pbar:
                model_pbar.set_postfix_str(model_name)

                try:
                    forecaster = _create_forecaster(
                        model_name,
                        {},
                        n_lags=n_lags,
                        random_seed=random_seed,
                        has_exog=exog_train is not None,
                        logger=self.logger,
                    )
                    search_space = _yaml_search_space_to_dict(model_search_spaces.get(model_name, {}))

                    # Match train: zero out imputed rows during fit.
                    if sample_mask_for_weights is not None:

                        def _weight_func(index, mask=sample_mask_for_weights):
                            return mask.reindex(index).fillna(False).astype(float).to_numpy()

                        forecaster.weight_func = _weight_func

                    tuning_results, _ = spotoptim_search_forecaster(
                        forecaster=forecaster,
                        y=y_train,
                        cv=cv,
                        search_space=search_space,
                        metric=channel_metric_callable,
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

                    per_model[model_name] = {
                        "best_metric": best_metric_val,
                        "best_params": best_params,
                        "best_lags": best_lags,
                    }

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
                    per_model[model_name] = {"error": str(e)}

            model_pbar.close()

            if best_overall is not None:
                results[target_col] = {**best_overall, "model_scores": per_model}
                self.logger.info(
                    f"  >> {target_col}: winner={best_overall['best_model']}, "
                    f"{metric}={best_overall['best_metric']:.6f}"
                )
            else:
                results[target_col] = {"error": "All models failed", "model_scores": per_model}

        channel_pbar.close()
        return results
