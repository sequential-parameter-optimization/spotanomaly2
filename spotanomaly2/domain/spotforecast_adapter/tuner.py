"""Per-panel hyperparameter tuning entry point: ``SpotforecastTuner``.

Wraps ``spotforecast2.model_selection.spotoptim_search_forecaster`` to run
surrogate-model Bayesian optimisation per target channel against every
candidate model in ``tune_config['models']``, picking the per-channel winner.

Mirrors ``SpotforecastTrainer.train_panel``'s preprocessing exactly (target/exog
split, imputation-flag handling, optional Ridge anomaly pre-pass, optional Δy
differentiation) so the tuned hyperparameters are optimal for the same objective
the trainer will actually fit. The shared preparation lives in ``channel_prep``
so the two entry points can't drift apart. R² under differentiation is reported
in raw-y space via ``_build_raw_r2_under_differentiation`` so the tuner
leaderboard matches what production scoring shows.
"""

from typing import Any

import pandas as pd

from spotanomaly2.infrastructure import logging

from .channel_prep import SKIP_CHANNEL, attach_weight_func, build_sample_mask, get_weight_suffix, prepare_panel
from .factory import _build_estimator, _create_forecaster
from .prediction import _difference
from .preprocessing import _compute_observed_mask, _ensure_freq, _interpolate_inplace
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
        from spotforecast2.model_selection import OneStepAheadFold
        from tqdm.auto import tqdm

        train_settings = self._train_settings()
        search_settings = self._search_settings(tune_config)

        # Match SpotforecastTrainer.train_panel exactly: include `exogenous_*`
        # columns as exog features and exclude them from the tuning targets,
        # then apply the same known-anomaly masking. Otherwise tune optimizes a
        # different model than train ends up fitting.
        weight_suffix = get_weight_suffix(self.config)
        df, target_cols, exog_columns = prepare_panel(self.config, df, weight_suffix, self.logger)

        if channels is not None:
            target_cols = [c for c in target_cols if c in channels]
            missing = set(channels) - set(target_cols)
            if missing:
                self.logger.warning(f"Requested channels not found in data: {missing}")

        # Use ALL the processed data — the trailing 10–20% carries the
        # strongest distribution drift; chopping it off would make CV blind
        # to live-shift. OneStepAheadFold below provides the train/val split.
        tune_df = _ensure_freq(df, self.config.get("process", {}).get("resample", {}).get("freq", "5min"))

        self.logger.info(
            f"Tuning panel {panel_id}: {len(target_cols)} channels, "
            f"{search_settings['n_trials']} trials, {search_settings['n_initial']} initial, "
            f"models={search_settings['models']}"
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
            channel_result = self._tune_channel(
                panel_id,
                target_col,
                tune_df,
                exog_columns,
                weight_suffix,
                cv,
                train_settings,
                search_settings,
                tune_config,
            )
            if channel_result is not None:
                results[target_col] = channel_result
        channel_pbar.close()
        return results

    # ------------------------------------------------------------------
    # Settings resolution
    # ------------------------------------------------------------------

    def _train_settings(self) -> dict[str, Any]:
        """Resolve the train-side knobs the tuner must mirror (lags, seed, Δy)."""
        train_cfg = self.config.get("train", {})
        n_lags = train_cfg.get("lags", 24)
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
        return {
            "n_lags": n_lags,
            "n_lags_max": n_lags_max,
            "random_seed": train_cfg.get("random_seed", 42),
            # Tune on Δy too — must match what train_panel fits, otherwise the
            # winning hyperparameters won't be optimal for the actual estimator
            # that ends up in production.
            "diff_order": int(train_cfg.get("differentiation", 1)),
        }

    def _search_settings(self, tune_config: dict[str, Any]) -> dict[str, Any]:
        """Resolve the SpotOptim search knobs (budget, metric, candidate models)."""
        metric = tune_config.get("metric", "mean_absolute_error")
        # Wrap the string metric so a single divergent MLP / Bayesian / Huber
        # trial doesn't crash SpotOptim's surrogate fit (GaussianProcessRegressor
        # rejects NaN/Inf). Falls through unchanged when metric is already a
        # callable or an unknown string.
        metric_callable = _build_nan_safe_metric(metric) if isinstance(metric, str) else metric

        models = tune_config.get("models", ["LightGBM"])
        if isinstance(models, str):
            models = [models]

        return {
            "n_trials": tune_config.get("n_trials", 10),
            "n_initial": tune_config.get("n_initial", 5),
            "metric": metric,
            "metric_callable": metric_callable,
            "models": self._validate_models(models),
            "model_search_spaces": tune_config.get("model_search_spaces", {}),
        }

    def _validate_models(self, models: list[str]) -> list[str]:
        """Keep only model names ``_build_estimator`` can construct.

        Log the exact cause so a typo or a missing dependency doesn't silently
        collapse the run to trees.
        """
        valid_models: list[str] = []
        for m in models:
            try:
                _build_estimator(m, {}, random_seed=0, logger=self.logger)
                valid_models.append(m)
            except ValueError as exc:
                self.logger.warning(f"Skipping model '{m}': {exc}")
            except ImportError as exc:
                self.logger.warning(f"Skipping model '{m}': dependency missing ({exc})")
        return valid_models or ["LightGBM"]

    def _channel_trial_budget(
        self,
        tune_config: dict[str, Any],
        panel_id: str,
        target_col: str,
        n_trials: int,
        n_initial: int,
    ) -> tuple[int, int]:
        """Apply per-panel / per-channel ``panel_overrides`` to the trial budget."""
        panel_overrides = tune_config.get("panel_overrides", {}).get(panel_id, {})
        if panel_overrides.get("default"):
            n_trials = panel_overrides["default"].get("n_trials", n_trials)
            n_initial = panel_overrides["default"].get("n_initial", n_initial)
        channel_overrides = panel_overrides.get("channels", {}).get(target_col, {})
        if channel_overrides:
            n_trials = channel_overrides.get("n_trials", n_trials)
            n_initial = channel_overrides.get("n_initial", n_initial)
        return n_trials, n_initial

    # ------------------------------------------------------------------
    # Per-channel tuning
    # ------------------------------------------------------------------

    def _tune_channel(
        self,
        panel_id: str,
        target_col: str,
        tune_df: pd.DataFrame,
        exog_columns: list[str],
        weight_suffix: str,
        cv,
        train_settings: dict[str, Any],
        search_settings: dict[str, Any],
        tune_config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Tune every candidate model for one channel; return the winner + scores.

        Returns ``None`` when the channel is skipped (insufficient data or all
        training samples masked out), otherwise a dict with the best result plus
        a ``model_scores`` sub-dict (or an ``{"error": ...}`` dict if every model
        failed).
        """
        n_lags = train_settings["n_lags"]
        n_lags_max = train_settings["n_lags_max"]
        random_seed = train_settings["random_seed"]
        diff_order = train_settings["diff_order"]

        # Match train: respect imputation `__weight` flags so imputed rows
        # don't masquerade as observations during the search. Targets are
        # already gap-free (process stage + _apply_known_anomaly_imputation).
        observed_mask = _compute_observed_mask(tune_df, target_col, weight_suffix)
        y_train = tune_df[target_col].copy()
        y_train.name = target_col
        if len(y_train) < n_lags_max + 10:
            self.logger.warning(f"  Skipping {target_col}: insufficient data ({len(y_train)} rows)")
            return None

        sample_mask = build_sample_mask(self.config, observed_mask, y_train, target_col, n_lags_max, self.logger)
        if sample_mask is SKIP_CHANNEL:
            self.logger.warning(f"  Skipping {target_col}: no fully observed training samples left")
            return None

        exog_train = None
        if exog_columns:
            exog_train = tune_df[exog_columns].loc[y_train.index]
            if exog_train.isna().any().any():
                exog_train = _interpolate_inplace(exog_train)

        # Predict Δy[t] just like train_panel will. The tuner must score
        # candidates on the same target the production model will fit
        # against, otherwise the winning hyperparameters are optimal
        # for a different objective.
        metric = search_settings["metric"]
        channel_metric_callable = search_settings["metric_callable"]
        if diff_order > 0:
            # Capture the raw (pre-difference) series — the R² metric
            # looks up actual raw values by timestamp inside the val
            # window so the leaderboard number matches production.
            y_train_raw_pre_diff = y_train.copy()
            y_train = _difference(y_train, diff_order).dropna()
            if exog_train is not None:
                exog_train = exog_train.loc[y_train.index]
            if sample_mask is not None:
                sample_mask = sample_mask.reindex(y_train.index).fillna(False)
            # Only R² needs the raw-space rescaling — MAE/MSE/RMSE
            # numerators are the same residuals in either space, so
            # they already match production without modification.
            if isinstance(metric, str) and metric.lower() in ("r2", "r2_score") and diff_order == 1:
                channel_metric_callable = _build_raw_r2_under_differentiation(y_train_raw_pre_diff)

        channel_n_trials, channel_n_initial = self._channel_trial_budget(
            tune_config, panel_id, target_col, search_settings["n_trials"], search_settings["n_initial"]
        )

        return self._search_models(
            target_col=target_col,
            y_train=y_train,
            exog_train=exog_train,
            sample_mask=sample_mask if isinstance(sample_mask, pd.Series) else None,
            cv=cv,
            channel_metric_callable=channel_metric_callable,
            metric=metric,
            models=search_settings["models"],
            model_search_spaces=search_settings["model_search_spaces"],
            n_trials=channel_n_trials,
            n_initial=channel_n_initial,
            n_lags=n_lags,
            random_seed=random_seed,
        )

    def _search_models(
        self,
        *,
        target_col: str,
        y_train: pd.Series,
        exog_train: pd.DataFrame | None,
        sample_mask: pd.Series | None,
        cv,
        channel_metric_callable,
        metric,
        models: list[str],
        model_search_spaces: dict[str, Any],
        n_trials: int,
        n_initial: int,
        n_lags: int | list[int],
        random_seed: int,
    ) -> dict[str, Any]:
        """Run the SpotOptim search for every candidate model and pick the winner."""
        from tqdm.auto import tqdm

        best_overall: dict[str, Any] | None = None
        per_model: dict[str, dict[str, Any]] = {}

        model_pbar = tqdm(models, desc=f"  {target_col}", unit="model", leave=False)
        for model_name in model_pbar:
            model_pbar.set_postfix_str(model_name)
            entry = self._search_one_model(
                target_col=target_col,
                model_name=model_name,
                y_train=y_train,
                exog_train=exog_train,
                sample_mask=sample_mask,
                cv=cv,
                channel_metric_callable=channel_metric_callable,
                metric=metric,
                raw_search_space=model_search_spaces.get(model_name, {}),
                n_trials=n_trials,
                n_initial=n_initial,
                n_lags=n_lags,
                random_seed=random_seed,
            )
            per_model[model_name] = entry
            if "error" not in entry:
                best_metric_val = entry["best_metric"]
                if best_overall is None or (
                    best_metric_val is not None and best_metric_val < best_overall["best_metric"]
                ):
                    best_overall = {
                        "best_params": entry["best_params"],
                        "best_lags": entry["best_lags"],
                        "best_metric": best_metric_val,
                        "best_model": model_name,
                    }
        model_pbar.close()

        if best_overall is not None:
            self.logger.info(
                f"  >> {target_col}: winner={best_overall['best_model']}, {metric}={best_overall['best_metric']:.6f}"
            )
            return {**best_overall, "model_scores": per_model}
        return {"error": "All models failed", "model_scores": per_model}

    def _search_one_model(
        self,
        *,
        target_col: str,
        model_name: str,
        y_train: pd.Series,
        exog_train: pd.DataFrame | None,
        sample_mask: pd.Series | None,
        cv,
        channel_metric_callable,
        metric,
        raw_search_space: dict[str, Any],
        n_trials: int,
        n_initial: int,
        n_lags: int | list[int],
        random_seed: int,
    ) -> dict[str, Any]:
        """Run one model's SpotOptim search; return its best result or an error dict."""
        from spotforecast2.model_selection import spotoptim_search_forecaster

        try:
            forecaster = _create_forecaster(
                model_name,
                {},
                n_lags=n_lags,
                random_seed=random_seed,
                has_exog=exog_train is not None,
                logger=self.logger,
            )
            search_space = _yaml_search_space_to_dict(raw_search_space)

            # Match train: zero out imputed rows during fit.
            attach_weight_func(forecaster, sample_mask)

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
                n_trials=n_trials,
                n_initial=n_initial,
                show_progress=False,
            )

            best_params, best_lags, best_metric_val = self._extract_best(tuning_results, n_lags)
            self.logger.info(f"    {target_col} [{model_name}]: {metric}={best_metric_val:.6f}, lags={best_lags}")
            return {
                "best_metric": best_metric_val,
                "best_params": best_params,
                "best_lags": best_lags,
            }
        except Exception as e:
            self.logger.error(
                f"  Tuning failed for {target_col} [{model_name}]: {e}",
                exc_info=True,
            )
            return {"error": str(e)}

    @staticmethod
    def _extract_best(tuning_results: pd.DataFrame, n_lags: int | list[int]) -> tuple[dict, Any, float | None]:
        """Pull ``(best_params, best_lags, best_metric)`` out of the SpotOptim leaderboard."""
        best_row = tuning_results.iloc[0]
        best_params = best_row["params"] if "params" in tuning_results.columns else {}
        best_lags = best_row["lags"] if "lags" in tuning_results.columns else n_lags

        non_meta_cols = [c for c in tuning_results.columns if c not in ("params", "lags") and c not in best_params]
        best_metric_val = None
        if non_meta_cols:
            raw_val = best_row[non_meta_cols[0]]
            if hasattr(raw_val, "__len__") and not isinstance(raw_val, str):
                best_metric_val = float(raw_val[0])
            else:
                best_metric_val = float(raw_val)
        return best_params, best_lags, best_metric_val
