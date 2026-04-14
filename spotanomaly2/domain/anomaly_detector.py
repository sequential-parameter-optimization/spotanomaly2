"""Anomaly detection service using spotforecast2 forecasting models."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from spotanomaly2_safe.scoring.pipeline import ForecastingAnomalyDetector

from spotanomaly2.domain.constants import MIN_TRAIN_SIZE, TRAIN_TEST_SPLIT_RATIO
from spotanomaly2.domain.exceptions import InsufficientDataException, ModelNotFoundException
from spotanomaly2.domain.spotforecast_adapter import SpotforecastTrainer
from spotanomaly2.infrastructure import logging, storage


class AnomalyDetector:
    """Service for detecting anomalies using forecast-based scoring."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("AnomalyDetector")
        self.model_timestamp = config.get("detect", {}).get("model_timestamp")

    def load_forecasting_model(self, panel_id: str) -> dict[str, Any]:
        """Load trained spotforecast2 model for a panel."""
        models_dir = Path(self.config["paths"]["models_dir"])
        fc_model_name = self.config["detect"]["fc_model_name"]

        model_filename = f"{fc_model_name}_fc_model_panel_{panel_id}.pkl"

        try:
            model_file = storage.find_latest_model(
                models_dir,
                model_filename,
                model_timestamp=self.model_timestamp,
            )
        except FileNotFoundError as e:
            self.logger.error(f"No model found: {e}")
            raise ModelNotFoundException(f"No model found: {e}") from e

        self.logger.info(f"Loading model from: {model_file}")

        model_data = joblib.load(model_file)
        if isinstance(model_data, dict) and "model_type" in model_data:
            return model_data

        raise ModelNotFoundException(f"Could not load model from {model_file}. Unknown format.")

    def build_anomaly_detector(self) -> ForecastingAnomalyDetector:
        """Build anomaly detector from configuration."""
        scorer_name = self.config["detect"]["scorer_name"]
        scorer_params = self.config["detect"]["scorer_params"].copy()
        high_quantile = self.config["detect"]["high_quantile"]

        normalize_scores = self.config["detect"].get("normalize_scores", True)
        normalization_quantile = self.config["detect"].get("normalization_quantile", 0.99)

        scorer_params.pop("window_agg", None)
        if "k" in scorer_params:
            scorer_params["n_clusters"] = scorer_params.pop("k")

        return ForecastingAnomalyDetector(
            scorer_name=scorer_name,
            scorer_params=scorer_params,
            high_quantile=high_quantile,
            normalize_scores=normalize_scores,
            normalization_quantile=normalization_quantile,
        )

    # ------------------------------------------------------------------
    # Test-window helpers
    # ------------------------------------------------------------------

    def _calculate_test_window_with_target_date(
        self, df: pd.DataFrame, target_date: str, hist_window: int
    ) -> tuple[int, int]:
        """Compute test window (start, end) when target_date is set."""
        target_dt = pd.to_datetime(target_date)
        self.logger.info(f"Using target date: {target_dt}")

        if isinstance(df.index, pd.DatetimeIndex):
            valid_indices = df.index <= target_dt
            if not valid_indices.any():
                raise ValueError(f"No data points found before or at target_date {target_dt}")
            target_idx = valid_indices[::-1].argmax()
            target_idx = len(df) - 1 - target_idx
        elif "timestamp" in df.columns:
            valid_indices = df["timestamp"] <= target_dt
            if not valid_indices.any():
                raise ValueError(f"No data points found before or at target_date {target_dt}")
            target_idx = valid_indices[::-1].argmax()
            target_idx = len(df) - 1 - target_idx
        else:
            raise ValueError("Cannot use target_date: no datetime index or timestamp column found")

        target_idx = min(target_idx, len(df) - 1)
        actual_target_time = (
            df.index[target_idx] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[target_idx]["timestamp"]
        )
        self.logger.info(f"Target date {target_dt} corresponds to index {target_idx} (timestamp: {actual_target_time})")
        test_end_idx = target_idx + 1
        test_start_idx = max(0, test_end_idx - hist_window)
        self.logger.info(
            f"Test window: index {test_start_idx} to {test_end_idx} (last timestamp: {actual_target_time})"
        )
        return test_start_idx, test_end_idx

    def _calculate_test_window_default(self, df: pd.DataFrame, hist_window: int) -> tuple[int, int]:
        """Compute test window (start, end) using end of time series."""
        test_end_idx = len(df)
        test_start_idx = max(0, test_end_idx - hist_window)
        self.logger.info("Using end of time series (no target_date specified)")
        self.logger.info(f"Test window: index {test_start_idx} to {test_end_idx}")
        return test_start_idx, test_end_idx

    def _adjust_window_for_insufficient_data(
        self,
        df: pd.DataFrame,
        test_start_idx: int,
        test_end_idx: int,
        hist_window: int,
    ) -> tuple[int, int]:
        """Validate and adjust scorer fit/eval window when data is insufficient."""
        n = len(df)
        min_required = hist_window * 2 + 1
        min_train_size = MIN_TRAIN_SIZE

        if n < min_required:
            if n < min_train_size * 2:
                raise InsufficientDataException(
                    f"Time series too short. "
                    f"Length: {n}, minimum required: {min_train_size * 2} samples "
                    f"(for {min_train_size} training + {min_train_size} testing)"
                )
            effective_train_size = max(min_train_size, int(n * TRAIN_TEST_SPLIT_RATIO))
            effective_test_size = max(min_train_size, n - effective_train_size)
            test_end_idx = len(df)
            test_start_idx = max(min_train_size, test_end_idx - effective_test_size)
            self.logger.warning(
                f"Insufficient data for full hist_window ({hist_window}). "
                f"Using adjusted window: {test_start_idx} samples for training, "
                f"{test_end_idx - test_start_idx} samples for testing."
            )
        elif test_start_idx < hist_window:
            if test_start_idx < min_train_size:
                raise InsufficientDataException(
                    f"Not enough data before test window for training. "
                    f"Required: {min_train_size} samples for training, "
                    f"available: {test_start_idx}"
                )
            remaining_data = len(df) - test_start_idx
            effective_test_size = min(hist_window, remaining_data)
            test_end_idx = min(len(df), test_start_idx + effective_test_size)
            self.logger.warning(
                f"Adjusting test window to use available training data. "
                f"Train size: {test_start_idx}, "
                f"Test size: {test_end_idx - test_start_idx}"
            )
        return test_start_idx, test_end_idx

    def _split_unseen_scoring_data(
        self,
        panel_id: str,
        df: pd.DataFrame,
        model_data: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (history_df, unseen_df) to avoid scorer leakage.

        The scorer must not be trained on rows that were used to train the
        forecasting model. This helper removes the forecaster training period
        from the scoring dataset.
        """
        if len(df) == 0:
            return df.iloc[0:0], df

        # Preferred: explicit timestamp boundary persisted with the model.
        train_end_timestamp = model_data.get("train_end_timestamp")
        if train_end_timestamp and isinstance(df.index, pd.DatetimeIndex):
            cutoff = pd.Timestamp(train_end_timestamp)
            if cutoff.tzinfo is None and df.index.tz is not None:
                cutoff = cutoff.tz_localize(df.index.tz)
            elif cutoff.tzinfo is not None and df.index.tz is None:
                cutoff = cutoff.tz_convert("UTC").tz_localize(None)

            history_df = df.loc[df.index <= cutoff]
            unseen_df = df.loc[df.index > cutoff]

            if len(unseen_df) == 0:
                raise InsufficientDataException(
                    f"Panel {panel_id}: no unseen rows after model train end "
                    f"({cutoff}). Need new data beyond training period for leakage-free scoring."
                )

            self.logger.info(
                f"Panel {panel_id}: leakage guard active using train_end={cutoff}. "
                f"Excluded {len(history_df)} seen row(s); {len(unseen_df)} unseen row(s) remain for scoring."
            )
            return history_df, unseen_df

        # Fallback for older model files without timestamp metadata.
        train_size = model_data.get("train_size")
        if isinstance(train_size, int) and 0 < train_size < len(df):
            history_df = df.iloc[:train_size]
            unseen_df = df.iloc[train_size:]
            self.logger.warning(
                f"Panel {panel_id}: model has no train_end_timestamp; using train_size={train_size} "
                "as leakage boundary fallback. Re-train models once to persist precise timestamps."
            )
            return history_df, unseen_df

        # Last-resort fallback using current config ratio.
        train_ratio = self.config.get("train", {}).get("train_ratio", 0.9)
        cutoff_pos = int(len(df) * float(train_ratio))
        cutoff_pos = max(1, min(cutoff_pos, len(df) - 1))
        history_df = df.iloc[:cutoff_pos]
        unseen_df = df.iloc[cutoff_pos:]
        self.logger.warning(
            f"Panel {panel_id}: model lacks training boundary metadata; "
            f"using ratio-based fallback ({train_ratio:.3f}). "
            "Re-train models to persist precise leakage boundary."
        )
        return history_df, unseen_df

    # ------------------------------------------------------------------
    # Flow weighting
    # ------------------------------------------------------------------

    def _apply_flow_weights(
        self,
        panel_id: str,
        source_fit_df: pd.DataFrame,
        source_eval_df: pd.DataFrame,
        fit_true_df: pd.DataFrame,
        fit_pred_df: pd.DataFrame,
        eval_true_df: pd.DataFrame,
        eval_pred_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Multiply true and pred values by exogenous flow so residuals are flow-weighted."""
        exog_cols = [c for c in source_fit_df.columns if c.startswith("exogenous_")]
        if not exog_cols:
            self.logger.warning(f"Panel {panel_id}: weight_residuals enabled but no exogenous columns found")
            return fit_true_df, fit_pred_df, eval_true_df, eval_pred_df

        # Use first exogenous column as flow weight (typically one per panel)
        flow_col = exog_cols[0]
        if len(exog_cols) > 1:
            self.logger.info(
                f"Panel {panel_id}: multiple exogenous columns found, using '{flow_col}' for flow weighting"
            )

        fit_flow = source_fit_df.loc[fit_true_df.index, flow_col]
        eval_flow = source_eval_df.loc[eval_true_df.index, flow_col]

        fit_true_df = fit_true_df.multiply(fit_flow, axis=0)
        fit_pred_df = fit_pred_df.multiply(fit_flow, axis=0)
        eval_true_df = eval_true_df.multiply(eval_flow, axis=0)
        eval_pred_df = eval_pred_df.multiply(eval_flow, axis=0)

        self.logger.info(
            f"Panel {panel_id}: applied flow weighting using '{flow_col}' to residuals"
        )

        return fit_true_df, fit_pred_df, eval_true_df, eval_pred_df

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _validate_scoring_inputs(
        self,
        panel_id: str,
        eval_true_df: pd.DataFrame,
        eval_pred_df: pd.DataFrame,
    ) -> None:
        """Validate scorer eval-window inputs and fail fast on NaN/Inf."""
        fail_on_nan_inputs = self.config.get("detect", {}).get("fail_on_nan_inputs", True)

        def _count_bad_values(df: pd.DataFrame) -> tuple[int, int]:
            arr = df.to_numpy()
            return int(np.isnan(arr).sum()), int(np.isinf(arr).sum())

        input_stats = {
            "eval_true": _count_bad_values(eval_true_df),
            "eval_pred": _count_bad_values(eval_pred_df),
        }
        total_bad = sum(nan + inf for nan, inf in input_stats.values())
        if total_bad == 0:
            return

        details = ", ".join(f"{name}(nan={nan}, inf={inf})" for name, (nan, inf) in input_stats.items())
        msg = f"Panel {panel_id}: invalid scoring inputs: {details}"
        if fail_on_nan_inputs:
            raise ValueError(msg)
        self.logger.warning(msg)

    def _exclude_imputed_rows(
        self,
        panel_id: str,
        window_name: str,
        source_df: pd.DataFrame,
        aligned_true_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Exclude rows where one or more scoring features are imputed.

        Uses existing imputation weight columns (suffix from config), where
        1 means observed value and 0 means imputed value.
        Handles both primary and exogenous sources.
        """
        weight_suffix = self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")

        # Find all weight columns for scoring features (both primary and exogenous).
        # For each weight column in source_df, check if the corresponding feature
        # (without suffix) is in aligned_true_df. This robustly handles cases where
        # some columns are not used in scoring.
        weight_cols = [
            col
            for col in source_df.columns
            if col.endswith(weight_suffix) and col[: -len(weight_suffix)] in aligned_true_df.columns
        ]

        if not weight_cols:
            keep_mask = pd.Series(True, index=aligned_true_df.index)
            return aligned_true_df, keep_mask

        weights = source_df.loc[aligned_true_df.index, weight_cols]
        # Conservative: treat missing weight entries as imputed (drop).
        observed_mask = (weights.fillna(0.0) >= 0.5).all(axis=1)

        dropped = int((~observed_mask).sum())
        if dropped > 0:
            self.logger.info(
                f"Panel {panel_id}: excluded {dropped} {window_name} row(s) with imputed values "
                f"based on {len(weight_cols)} weight column(s) (primary + exogenous)"
            )

        return aligned_true_df.loc[observed_mask], observed_mask

    def _exclude_invalid_scorer_fit_rows(
        self,
        panel_id: str,
        fit_true_df: pd.DataFrame,
        fit_pred_df: pd.DataFrame,
        eval_true_df: pd.DataFrame,
        eval_pred_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Exclude invalid rows from the scorer-fit window.

        Drops rows where the scorer-fit matrices contain NaN/Inf while
        keeping the eval window intact.
        """
        common_cols = [
            c
            for c in fit_true_df.columns
            if c in fit_pred_df.columns and c in eval_true_df.columns and c in eval_pred_df.columns
        ]
        if not common_cols:
            raise ValueError(f"Panel {panel_id}: no common columns for scoring")

        fit_true_df = fit_true_df[common_cols]
        fit_pred_df = fit_pred_df[common_cols]
        eval_true_df = eval_true_df[common_cols]
        eval_pred_df = eval_pred_df[common_cols]

        mask_true = np.isfinite(fit_true_df.to_numpy()).all(axis=1)
        mask_pred = np.isfinite(fit_pred_df.to_numpy()).all(axis=1)
        fit_keep_mask = mask_true & mask_pred

        dropped_rows = int((~fit_keep_mask).sum())
        if dropped_rows > 0:
            dropped_index = fit_true_df.index[~fit_keep_mask]
            first_ts = dropped_index.min()
            last_ts = dropped_index.max()
            self.logger.warning(
                f"Panel {panel_id}: excluding {dropped_rows} invalid scorer-fit row(s) "
                f"({first_ts} to {last_ts})"
            )

        fit_true_df = fit_true_df.loc[fit_keep_mask]
        fit_pred_df = fit_pred_df.loc[fit_keep_mask]

        if len(fit_true_df) == 0:
            raise ValueError(f"Panel {panel_id}: no valid rows left in scorer-fit window after exclusion")

        return fit_true_df, fit_pred_df, eval_true_df, eval_pred_df

    def detect_panel(
        self, panel_id: str, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """Detect anomalies for a single panel."""
        self.logger.info(f"Detecting anomalies for panel {panel_id}...")

        model_data = self.load_forecasting_model(panel_id)
        model_type = model_data.get("model_type", "unknown")

        history_df, unseen_df = self._split_unseen_scoring_data(
            panel_id=panel_id,
            df=df,
            model_data=model_data,
        )

        hist_window = self.config["detect"]["hist_window"]
        target_date = self.config["detect"].get("target_date", None)

        if target_date:
            test_start_idx, test_end_idx = (
                self._calculate_test_window_with_target_date(
                    unseen_df, target_date, hist_window
                )
            )
        else:
            test_start_idx, test_end_idx = self._calculate_test_window_default(
                unseen_df, hist_window
            )

        test_start_idx, test_end_idx = self._adjust_window_for_insufficient_data(
            unseen_df, test_start_idx, test_end_idx, hist_window
        )

        # Split unseen data into two windows:
        #   scorer_fit_df  — scorer learns "normal residual" distribution here
        #   scorer_eval_df — scorer flags anomalies here
        # Both windows are AFTER the forecaster's training period (leakage guard),
        # so all predictions are genuine out-of-sample.
        scorer_fit_df = unseen_df.iloc[:test_start_idx]
        scorer_eval_df = unseen_df.iloc[test_start_idx:test_end_idx]

        self.logger.info(f"Scorer fit window: {len(scorer_fit_df)}, Scorer eval window: {len(scorer_eval_df)}")

        # Generate predictions via spotforecast2 adapter
        self.logger.info(f"Generating predictions using {model_type} model...")
        adapter = SpotforecastTrainer(self.config, self.logger)
        history_for_fit_pred = history_df if len(history_df) > 0 else None
        fit_pred_df = adapter.predict(
            model_data,
            scorer_fit_df,
            history_df=history_for_fit_pred,
        )
        eval_history_df = pd.concat([history_df, scorer_fit_df]) if len(history_df) > 0 else scorer_fit_df
        eval_pred_df = adapter.predict(model_data, scorer_eval_df, history_df=eval_history_df)

        # Align true values with predictions
        target_cols = fit_pred_df.columns
        fit_true_df = scorer_fit_df.loc[fit_pred_df.index, target_cols]
        eval_true_df = scorer_eval_df.loc[eval_pred_df.index, target_cols]

        fit_true_df, fit_keep_mask = self._exclude_imputed_rows(
            panel_id=panel_id,
            window_name="scorer_fit",
            source_df=scorer_fit_df,
            aligned_true_df=fit_true_df,
        )
        fit_pred_df = fit_pred_df.loc[fit_keep_mask]

        eval_true_df, eval_keep_mask = self._exclude_imputed_rows(
            panel_id=panel_id,
            window_name="scorer_eval",
            source_df=scorer_eval_df,
            aligned_true_df=eval_true_df,
        )
        eval_pred_df = eval_pred_df.loc[eval_keep_mask]

        if len(fit_true_df) == 0 or len(eval_true_df) == 0:
            raise ValueError(f"Panel {panel_id}: no rows left after excluding imputed values via weight columns")

        fit_true_df, fit_pred_df, eval_true_df, eval_pred_df = self._exclude_invalid_scorer_fit_rows(
            panel_id=panel_id,
            fit_true_df=fit_true_df,
            fit_pred_df=fit_pred_df,
            eval_true_df=eval_true_df,
            eval_pred_df=eval_pred_df,
        )

        # Keep unweighted predictions for report visualizations
        report_pred_df = eval_pred_df.copy()

        # Flow-weight residuals: multiply both true and pred by exogenous flow
        # so that residual = flow*(actual - predicted). This suppresses anomalies
        # during low-flow periods and amplifies them during high-flow periods.
        weight_cfg = self.config.get("exogenous", {}).get("weight_residuals", {})
        if weight_cfg.get("enabled", False):
            fit_true_df, fit_pred_df, eval_true_df, eval_pred_df = self._apply_flow_weights(
                panel_id=panel_id,
                source_fit_df=scorer_fit_df,
                source_eval_df=scorer_eval_df,
                fit_true_df=fit_true_df,
                fit_pred_df=fit_pred_df,
                eval_true_df=eval_true_df,
                eval_pred_df=eval_pred_df,
            )

        self._validate_scoring_inputs(
            panel_id=panel_id,
            eval_true_df=eval_true_df,
            eval_pred_df=eval_pred_df,
        )

        if len(fit_pred_df) == 0:
            raise InsufficientDataException(
                f"Panel {panel_id}: no valid samples in scorer-fit window after prediction. "
                f"This usually means one or more target columns are entirely NaN "
                f"in the scorer-fit window. "
                f"Fix: delete data/processed/live/ so it re-bootstraps from the "
                f"clean baseline, then restart live mode."
            )

        # Build anomaly detector and run scoring
        self.logger.info("Building anomaly detector...")
        detector = self.build_anomaly_detector()

        self.logger.info("Computing anomaly scores and detecting anomalies...")
        contributions_df: pd.DataFrame | None = None
        try:
            scores_df, flags_df = detector.fit_score_detect(
                y_true_train=fit_true_df,
                y_pred_train=fit_pred_df,
                y_true_test=eval_true_df,
                y_pred_test=eval_pred_df,
            )
        except ValueError as exc:
            self.logger.warning(
                f"Panel {panel_id}: scoring failed ({exc}); returning NaN scores and no flags for this panel"
            )
            scores_df = pd.DataFrame(
                {
                    "anomaly_score": np.nan,
                    "anomaly_score_normalized": np.nan,
                },
                index=eval_true_df.index,
            )
            flags_df = pd.DataFrame(
                {
                    "anomaly_flag": 0,
                },
                index=eval_true_df.index,
            )

        if scores_df.index.tz is None:
            scores_df.index = scores_df.index.tz_localize("UTC")
            flags_df.index = flags_df.index.tz_localize("UTC")
            report_pred_df.index = report_pred_df.index.tz_localize("UTC")
            if contributions_df is not None:
                contributions_df.index = contributions_df.index.tz_localize("UTC")

        num_anomalies = flags_df.sum().sum()
        self.logger.info(f"Detected {num_anomalies} anomalies for panel {panel_id}")

        return scores_df, flags_df, report_pred_df, contributions_df

    def detect_all_panels(
        self, panel_data: dict[str, pd.DataFrame]
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]]:
        results = {}
        for panel_id, df in panel_data.items():
            self.logger.info(f"Detecting anomalies for panel {panel_id}...")
            result = self.detect_panel(panel_id, df)
            results[panel_id] = result
            self.logger.info(f"Completed detection for panel {panel_id}")
        return results

    def run(
        self, panel_data: dict[str, pd.DataFrame]
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]]:
        self.logger.info("Starting anomaly detection...")
        results = self.detect_all_panels(panel_data)
        self.logger.info("Anomaly detection completed successfully")
        return results
