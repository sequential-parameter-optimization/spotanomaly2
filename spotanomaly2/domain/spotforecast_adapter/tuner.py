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

from spotanomaly2.application.config import resolve_data_split
from spotanomaly2.infrastructure import logging

from .channel_prep import (
    TrainKnobs,
    prepare_channel,
    prepare_panel,
    resolve_train_settings,
)
from .factory import _build_estimator, _create_forecaster
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
        from spotforecast2_safe.splitter import OneStepAheadFold
        from tqdm.auto import tqdm

        train_settings = resolve_train_settings(self.config)
        split = resolve_data_split(self.config)
        search_settings = self._search_settings(tune_config)

        # Match SpotforecastTrainer.train_panel exactly: include `exogenous_*`
        # columns as exog features and exclude them from the tuning targets,
        # then apply the same known-anomaly masking. Otherwise tune optimizes a
        # different model than train ends up fitting.
        weight_suffix = train_settings.weight_suffix
        df, target_cols, exog_columns = prepare_panel(self.config, df, weight_suffix, self.logger)

        # Carve off the score% rows BEFORE CV — they are the scorer's territory
        # and must never influence hyperparameter selection. The tuner CV's
        # available window is the first ``train + test`` percentages of data.
        n_full = len(df)
        tune_available = int(n_full * (split.train + split.test) / 100)
        df = df.iloc[:tune_available]

        if channels is not None:
            target_cols = [c for c in target_cols if c in channels]
            missing = set(channels) - set(target_cols)
            if missing:
                self.logger.warning(f"Requested channels not found in data: {missing}")

        # Use ALL the processed data — the trailing 10–20% carries the
        # strongest distribution drift; chopping it off would make CV blind
        # to live-shift. OneStepAheadFold below provides the train/val split.
        # ``prepare_panel`` already applied ``_ensure_freq``.
        tune_df = df

        self.logger.info(
            f"Tuning panel {panel_id}: {len(target_cols)} channels, "
            f"{search_settings['n_trials']} trials, {search_settings['n_initial']} initial, "
            f"models={search_settings['models']}"
        )

        # OneStepAheadFold's train/val boundary lines up with the trainer's
        # train/test boundary: CV-train on the first ``train`` rows, CV-val
        # on the ``test`` rows. The ``score`` rows were already sliced off
        # above, so the tuner physically cannot see them — closes the
        # hyperparameter-selection leakage the old ``train_ratio`` design had.
        cv_train_size = max(1, int(n_full * split.train / 100))
        cross_validator = OneStepAheadFold(initial_train_size=cv_train_size, verbose=False)

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
                cross_validator,
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

    def _search_settings(self, tune_config: dict[str, Any]) -> dict[str, Any]:
        """Resolve the SpotOptim search knobs (budget, metric, candidate models)."""
        metric = tune_config.get("metric", "mean_absolute_error")
        # Wrap the string metric so a single divergent MLP / Bayesian / Huber
        # trial doesn't crash SpotOptim's surrogate fit (GaussianProcessRegressor
        # rejects NaN/Inf). Falls through unchanged when metric is already a
        # callable or an unknown string.
        default_metric_callable = _build_nan_safe_metric(metric) if isinstance(metric, str) else metric

        models = tune_config.get("models", ["LightGBM"])
        if isinstance(models, str):
            models = [models]

        return {
            "n_trials": tune_config.get("n_trials", 10),
            "n_initial": tune_config.get("n_initial", 5),
            "metric": metric,
            # May be overridden per channel (see ``_tune_channel`` for the
            # raw-y-space R² override under differentiation).
            "default_metric_callable": default_metric_callable,
            "models": self._validate_models(models),
            "model_search_spaces": tune_config.get("model_search_spaces", {}),
        }

    def _validate_models(self, models: list[str]) -> list[str]:
        """Keep only model names ``_build_estimator`` can construct.

        Fails loud if every requested model is invalid: silently collapsing a
        4-model search down to LightGBM-only because of an ``ImportError``
        would produce a leaderboard that doesn't match what the user asked for.
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
        if not valid_models:
            raise ValueError(
                f"No valid models in tune.models. Requested: {models}. See warnings above for per-model reasons."
            )
        return valid_models

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
        cross_validator,
        train_settings: TrainKnobs,
        search_settings: dict[str, Any],
        tune_config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Tune every candidate model for one channel; return the winner + scores.

        Returns ``None`` when the channel is skipped (insufficient data or all
        training samples masked out), otherwise a dict with the best result plus
        a ``model_scores`` sub-dict (or an ``{"error": ...}`` dict if every model
        failed).
        """
        n_lags = train_settings.fallback_lags
        n_lags_max = train_settings.fallback_lags_max
        random_seed = train_settings.random_seed
        diff_order = train_settings.diff_order

        # Shared per-channel setup so the tuner scores candidates on the exact
        # inputs train_panel will fit: imputation-flag mask, exog imputation,
        # and Δy. ``n_lags_max`` (search-space upper bound) bounds the mask
        # width. ``None`` => skip this channel.
        channel = prepare_channel(
            self.config,
            tune_df,
            target_col,
            exog_columns,
            weight_suffix,
            n_lags_for_mask=n_lags_max,
            diff_order=diff_order,
            logger=self.logger,
        )
        if channel is None:
            return None
        y_train = channel.y_fit
        exog_train = channel.exog_fit
        sample_mask = channel.sample_mask

        # Under differentiation only R² needs rescaling back to raw-y space so
        # the leaderboard number matches production — MAE/MSE/RMSE residuals are
        # identical in Δ and raw space. The metric looks up raw values by
        # timestamp inside the val window via the pre-difference series.
        metric = search_settings["metric"]
        channel_metric_callable = search_settings["default_metric_callable"]
        if diff_order == 1 and isinstance(metric, str) and metric.lower() in ("r2", "r2_score"):
            channel_metric_callable = _build_raw_r2_under_differentiation(channel.y_raw)

        channel_n_trials, channel_n_initial = self._channel_trial_budget(
            tune_config, panel_id, target_col, search_settings["n_trials"], search_settings["n_initial"]
        )

        return self._search_models(
            target_col=target_col,
            y_train=y_train,
            exog_train=exog_train,
            sample_mask=sample_mask,
            cv=cross_validator,
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

            # Imputed/anomalous-row exclusion is train-only (audit C2): SpotforecastTrainer
            # applies it via forecaster.weight_func, which forecaster.fit honours. The
            # one-step-ahead tuning objective fits the bare estimator and ignores
            # weight_func (and spotforecast2 rejects NaN targets), so it cannot drop
            # individual rows here. sample_mask still gates channel-skipping in
            # prepare_channel, keeping tuning and training aligned on trainable channels.

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
