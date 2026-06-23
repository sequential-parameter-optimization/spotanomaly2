"""Anomaly detection service using spotforecast2 forecasting models."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from spotanomaly2_safe.scoring.pipeline import ForecastingAnomalyDetector

from spotanomaly2.domain.constants import MIN_TRAIN_SIZE, TRAIN_TEST_SPLIT_RATIO
from spotanomaly2.domain.exceptions import InsufficientDataException, ModelNotFoundException
from spotanomaly2.domain.exogenous.residual_multiplier import find_multiplier_column, multiplier_prefixes
from spotanomaly2.domain.spotforecast_adapter import SpotforecastPredictor
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

        model_filename = f"fc_model_panel_{panel_id}.pkl"

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

    def build_anomaly_detector(self, high_quantile: float | None = None) -> ForecastingAnomalyDetector:
        """Build anomaly detector from configuration.

        Args:
            high_quantile: Detection quantile to use. When ``None`` (default) the
                configured ``detect.high_quantile`` is used. Group scoring passes
                a Bonferroni-corrected quantile here (see :meth:`_score_grouped`).
        """
        scorer_name = self.config["detect"]["scorer_name"]
        scorer_params = self.config["detect"]["scorer_params"].copy()
        if high_quantile is None:
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

    def _resolve_scorer_fit_scope(self) -> str:
        """Resolve which rows the scorer is allowed to learn its baseline from.

        Controlled by ``detect.scorer_fit_scope``:

        - ``"unseen"`` (default): leakage-guarded. The scorer fits only on data
          held out from BOTH the forecaster's training and the tuner's
          hyperparameter selection — i.e. the configured ``train.split.test``
          window. Forecast residuals there are genuine out-of-sample, so the
          learned "normal" distribution is unbiased and detection rates are
          honest.
        - ``"all"``: the scorer fits on the entire dataset (train + val + test).
          Residuals over the forecaster's training rows are optimistically
          small (in-sample fit), which shrinks the learned "normal" spread and
          can mask true anomalies. Use for comparison/diagnostics only, not
          production monitoring.

        ``"test"`` and ``"held_out"`` are accepted aliases for ``"unseen"``;
        ``"full"`` is an alias for ``"all"``.
        """
        raw = str(self.config.get("detect", {}).get("scorer_fit_scope", "unseen")).strip().lower()
        if raw in ("unseen", "test", "held_out", "held-out"):
            return "unseen"
        if raw in ("all", "full"):
            return "all"
        raise ValueError(f"detect.scorer_fit_scope must be 'unseen' or 'all', got {raw!r}")

    def _split_unseen_scoring_data(
        self,
        panel_id: str,
        df: pd.DataFrame,
        model_data: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (history_df, unseen_df) to avoid scorer leakage.

        The scorer must not be fit on rows the trainer OR tuner saw, otherwise
        hyperparameters chosen on those rows bias the residual distribution
        the scorer learns. The boundary the scorer cares about is the start of
        the configured ``train.split.test`` window — everything before is
        "seen by the pipeline", everything from there on is "unseen".

        When ``detect.scorer_fit_scope`` is ``"all"`` the leakage guard is
        disabled: the whole dataset is returned as "unseen" so the scorer can
        learn from train + val + test data (see :meth:`_resolve_scorer_fit_scope`).

        Lookup precedence (``"unseen"`` scope only):
          1. ``test_start_timestamp`` (preferred — written by current trainer)
          2. ``train_end_timestamp`` (legacy — points at the old train/test
             boundary; less strict because the tuner also CV'd on val%, but
             still the best signal an older artifact carries)
          3. ``train_size`` (older still)
          4. Config-based fallback using ``train.split`` percentages.
        """
        if len(df) == 0:
            return df.iloc[0:0], df

        # Scope "all": skip the leakage guard entirely and let the scorer fit on
        # every row. history_df is empty so detect_panel treats the full dataset
        # as the scorer-fit/eval pool.
        if self._resolve_scorer_fit_scope() == "all":
            self.logger.warning(
                f"Panel {panel_id}: detect.scorer_fit_scope='all' — leakage guard DISABLED. "
                f"Scorer fits on all {len(df)} row(s), including forecaster-training data. "
                f"The normal-residual baseline is optimistically biased; use for "
                f"comparison/diagnostics only, not production monitoring."
            )
            return df.iloc[0:0], df

        # Preferred: explicit test-window boundary persisted with the model.
        test_start_timestamp = model_data.get("test_start_timestamp")
        if test_start_timestamp and isinstance(df.index, pd.DatetimeIndex):
            cutoff = self._align_timestamp_to_index(pd.Timestamp(test_start_timestamp), df)
            history_df = df.loc[df.index < cutoff]
            unseen_df = df.loc[df.index >= cutoff]
            if len(unseen_df) == 0:
                raise InsufficientDataException(
                    f"Panel {panel_id}: no unseen rows at or after the test boundary "
                    f"({cutoff}). Need new data beyond the test window for scoring."
                )
            self.logger.info(
                f"Panel {panel_id}: leakage guard active using test_start={cutoff}. "
                f"Excluded {len(history_df)} pipeline-seen row(s); "
                f"{len(unseen_df)} unseen row(s) remain for scoring."
            )
            return history_df, unseen_df

        # Legacy: older artifacts predate the held-out test split; the strictest
        # boundary they carry is the trainer's old train_end.
        train_end_timestamp = model_data.get("train_end_timestamp")
        if train_end_timestamp and isinstance(df.index, pd.DatetimeIndex):
            cutoff = self._align_timestamp_to_index(pd.Timestamp(train_end_timestamp), df)
            history_df = df.loc[df.index <= cutoff]
            unseen_df = df.loc[df.index > cutoff]
            if len(unseen_df) == 0:
                raise InsufficientDataException(
                    f"Panel {panel_id}: no unseen rows after model train end "
                    f"({cutoff}). Need new data beyond training period for leakage-free scoring."
                )
            self.logger.warning(
                f"Panel {panel_id}: model has no test_start_timestamp; using legacy "
                f"train_end={cutoff} as boundary. Re-train to get the stricter split-based boundary."
            )
            return history_df, unseen_df

        train_size = model_data.get("train_size")
        if isinstance(train_size, int) and 0 < train_size < len(df):
            history_df = df.iloc[:train_size]
            unseen_df = df.iloc[train_size:]
            self.logger.warning(
                f"Panel {panel_id}: model has no timestamp metadata; using train_size={train_size} "
                "as leakage boundary fallback. Re-train models to persist precise timestamps."
            )
            return history_df, unseen_df

        # Last-resort: derive from current config's train.split.
        from spotanomaly2.application.config import resolve_data_split

        split = resolve_data_split(self.config)
        cutoff_pos = int(len(df) * (split.train + split.val) / 100)
        cutoff_pos = max(1, min(cutoff_pos, len(df) - 1))
        history_df = df.iloc[:cutoff_pos]
        unseen_df = df.iloc[cutoff_pos:]
        self.logger.warning(
            f"Panel {panel_id}: model lacks training boundary metadata; "
            f"using config split fallback (train={split.train}%, val={split.val}%). "
            "Re-train models to persist precise leakage boundary."
        )
        return history_df, unseen_df

    @staticmethod
    def _align_timestamp_to_index(ts: pd.Timestamp, df: pd.DataFrame) -> pd.Timestamp:
        """Align a persisted timestamp's tz to ``df.index`` for safe comparison."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return ts
        if ts.tzinfo is None and df.index.tz is not None:
            return ts.tz_localize(df.index.tz)
        if ts.tzinfo is not None and df.index.tz is None:
            return ts.tz_convert("UTC").tz_localize(None)
        return ts

    # ------------------------------------------------------------------
    # Flow weighting
    # ------------------------------------------------------------------

    def _apply_residual_multiplier(
        self,
        panel_id: str,
        multiplier_col: str,
        source_fit_df: pd.DataFrame,
        source_eval_df: pd.DataFrame,
        fit_true_df: pd.DataFrame,
        fit_pred_df: pd.DataFrame,
        eval_true_df: pd.DataFrame,
        eval_pred_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Multiply true and pred values by *multiplier_col* so residuals scale with it."""
        fit_mult = source_fit_df.loc[fit_true_df.index, multiplier_col]
        eval_mult = source_eval_df.loc[eval_true_df.index, multiplier_col]

        fit_true_df = fit_true_df.multiply(fit_mult, axis=0)
        fit_pred_df = fit_pred_df.multiply(fit_mult, axis=0)
        eval_true_df = eval_true_df.multiply(eval_mult, axis=0)
        eval_pred_df = eval_pred_df.multiply(eval_mult, axis=0)

        self.logger.info(f"Panel {panel_id}: multiplied residuals by '{multiplier_col}'")

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
                f"Panel {panel_id}: excluding {dropped_rows} invalid scorer-fit row(s) ({first_ts} to {last_ts})"
            )

        fit_true_df = fit_true_df.loc[fit_keep_mask]
        fit_pred_df = fit_pred_df.loc[fit_keep_mask]

        if len(fit_true_df) == 0:
            raise ValueError(f"Panel {panel_id}: no valid rows left in scorer-fit window after exclusion")

        return fit_true_df, fit_pred_df, eval_true_df, eval_pred_df

    # ------------------------------------------------------------------
    # Combined scoring (system-wide or per group)
    # ------------------------------------------------------------------

    @staticmethod
    def _bonferroni_quantile(high_quantile: float, n_groups: int) -> float:
        """Bonferroni-correct a system-level detection quantile for ``n_groups``.

        Splitting the scored channels into groups and flagging when *any* group
        trips (logical OR) multiplies the chances to raise a false alarm by the
        number of groups. To keep the system-wide false-positive rate at the
        configured ``1 - high_quantile``, the tail mass allotted to each group
        must be that budget divided across the groups::

            per_group_quantile = 1 - (1 - high_quantile) / n_groups

        e.g. ``high_quantile=0.999`` with 2 groups → ``0.9995`` per group.
        """
        if n_groups <= 1:
            return high_quantile
        return 1.0 - (1.0 - high_quantile) / n_groups

    def _resolve_groups(self, panel_id: str, columns: list[str]) -> dict[str, dict] | None:
        """Resolve ``detect.groups`` for *panel_id* against the scored *columns*.

        ``detect.groups`` is keyed per panel. Each group is either a plain list of
        channels (which uses the Bonferroni-corrected quantile) or a mapping with
        ``channels`` and an optional ``quantile`` override::

            detect:
              groups:
                "1":
                  chem: [channel_1_ph, channel_3_turbidity]          # Bonferroni default
                  physical:
                    channels: [channel_1_orp_mv, channel_1_temperature]
                    quantile: 0.999                                  # explicit override

        Each group's channels are intersected with *columns*. Columns that appear
        in no group are excluded from scoring (logged). Returns
        ``{group_name: {"columns": [...], "quantile": float | None}}`` (``quantile``
        is ``None`` when the group should use the Bonferroni default), or ``None``
        when no groups apply to this panel (caller scores all channels as one system).
        """
        groups_cfg = self.config.get("detect", {}).get("groups")
        if not groups_cfg:
            return None

        panel_groups = groups_cfg.get(str(panel_id))
        if not panel_groups:
            self.logger.info(f"Panel {panel_id}: no groups configured; scoring all channels as one system")
            return None

        resolved: dict[str, dict] = {}
        assigned: set[str] = set()
        for name, spec in panel_groups.items():
            cols, quantile = self._parse_group_spec(panel_id, name, spec)
            present = [c for c in cols if c in columns]
            missing = [c for c in cols if c not in columns]
            if missing:
                self.logger.warning(
                    f"Panel {panel_id}: group '{name}' lists column(s) not present in scored data: {missing}"
                )
            if not present:
                self.logger.warning(f"Panel {panel_id}: group '{name}' has no present columns; skipping group")
                continue
            resolved[name] = {"columns": present, "quantile": quantile}
            assigned.update(present)

        ungrouped = [c for c in columns if c not in assigned]
        if ungrouped:
            self.logger.warning(
                f"Panel {panel_id}: {len(ungrouped)} channel(s) not in any group — excluded from scoring: {ungrouped}"
            )

        if not resolved:
            self.logger.warning(
                f"Panel {panel_id}: no usable groups after resolution; scoring all channels as one system"
            )
            return None
        return resolved

    @staticmethod
    def _parse_group_spec(panel_id: str, name: str, spec: Any) -> tuple[list[str], float | None]:
        """Parse one group spec into ``(channels, quantile_override)``.

        Accepts a plain list of channel names (no override → ``quantile=None``) or
        a mapping with ``channels`` (alias ``columns``) and an optional ``quantile``.
        """
        if isinstance(spec, dict):
            cols = spec.get("channels", spec.get("columns"))
            if not isinstance(cols, list):
                raise ValueError(f"Panel {panel_id}: group '{name}' must define a 'channels' list; got {spec!r}")
            quantile = spec.get("quantile")
            if quantile is not None:
                quantile = float(quantile)
                if not 0.0 < quantile < 1.0:
                    raise ValueError(f"Panel {panel_id}: group '{name}' quantile must be in (0, 1); got {quantile}")
            return list(cols), quantile
        if isinstance(spec, list):
            return list(spec), None
        raise ValueError(
            f"Panel {panel_id}: group '{name}' must be a list of channels or a mapping with 'channels'; got {spec!r}"
        )

    def _anchor_normalized_scores(self, detector: ForecastingAnomalyDetector, scores_df: pd.DataFrame) -> None:
        """Rescale ``anomaly_score_normalized`` so 1.0 lines up with the flag threshold.

        Two upstream issues motivate this:
          1. ``ForecastingAnomalyDetector.score_and_detect()`` refits the
             normalizer on the *test* window, which collapses the range in
             short/quiet live batches and pushes ordinary scores to ~1.0.
          2. Even when refit on the train window, the normalizer saturates at
             ``q_high(train) + 20%`` — unrelated to the detection threshold.

        Anchoring at the train-fitted detection threshold makes
        ``normalized == 1.0`` iff the point is at/above the anomaly flag.
        """
        train_threshold = getattr(getattr(detector, "detector", None), "threshold", None)
        train_q_low = getattr(getattr(detector, "normalizer", None), "q_low", None)
        if (
            "anomaly_score_normalized" in scores_df.columns
            and train_threshold is not None
            and train_q_low is not None
            and train_threshold > train_q_low
        ):
            raw = scores_df["anomaly_score"].to_numpy()
            normalized = (raw - train_q_low) / (train_threshold - train_q_low)
            scores_df["anomaly_score_normalized"] = np.clip(normalized, 0.0, 1.0)

    def _score_one(
        self,
        panel_id: str,
        label: str,
        fit_true_df: pd.DataFrame,
        fit_pred_df: pd.DataFrame,
        eval_true_df: pd.DataFrame,
        eval_pred_df: pd.DataFrame,
        high_quantile: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit/score/detect a single scorer over the given columns at *high_quantile*.

        Returns ``(scores_df, flags_df)`` with the normalized score anchored to the
        flag threshold. On a scoring ``ValueError`` returns NaN scores and no flags
        (so a single failing group/system doesn't abort the panel).
        """
        detector = self.build_anomaly_detector(high_quantile=high_quantile)
        try:
            scores_df, flags_df = detector.fit_score_detect(
                y_true_train=fit_true_df,
                y_pred_train=fit_pred_df,
                y_true_test=eval_true_df,
                y_pred_test=eval_pred_df,
            )
            self._anchor_normalized_scores(detector, scores_df)
            # Expose the train-fitted raw flag threshold so callers can plot/compare
            # raw scores against the line where the flag fires (additive column).
            train_threshold = getattr(getattr(detector, "detector", None), "threshold", None)
            scores_df["anomaly_threshold"] = float(train_threshold) if train_threshold is not None else np.nan
            return scores_df, flags_df
        except ValueError as exc:
            self.logger.warning(f"Panel {panel_id}: scoring {label} failed ({exc}); returning NaN scores and no flags")
            scores_df = pd.DataFrame(
                {"anomaly_score": np.nan, "anomaly_score_normalized": np.nan, "anomaly_threshold": np.nan},
                index=eval_true_df.index,
            )
            flags_df = pd.DataFrame({"anomaly_flag": 0}, index=eval_true_df.index)
            return scores_df, flags_df

    def _score_grouped(
        self,
        panel_id: str,
        groups: dict[str, dict],
        fit_true_df: pd.DataFrame,
        fit_pred_df: pd.DataFrame,
        eval_true_df: pd.DataFrame,
        eval_pred_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score each group independently and OR their flags into a system flag.

        Each group is scored by its own scorer over only that group's channels.
        A group without an explicit ``quantile`` uses the Bonferroni-corrected
        quantile (so the OR of the per-group flags keeps the configured
        system-wide false-positive rate); a group with a ``quantile`` override
        uses that value directly (trading the budget guarantee for a hand-tuned
        sensitivity). The returned ``flags_df['anomaly_flag']`` is 1 iff at least
        one group flagged; ``scores_df`` carries the system score (max normalized
        across groups, with the matching raw score) plus per-group columns.
        """
        base_q = self.config["detect"]["high_quantile"]
        default_q = self._bonferroni_quantile(base_q, len(groups))
        applied_q = {name: (g["quantile"] if g["quantile"] is not None else default_q) for name, g in groups.items()}
        overrides = {name: g["quantile"] for name, g in groups.items() if g["quantile"] is not None}
        self.logger.info(
            f"Panel {panel_id}: scoring {len(groups)} group(s) {list(groups)}; "
            f"Bonferroni default {base_q} → {default_q} per group" + (f"; overrides {overrides}" if overrides else "")
        )

        norm_by_group: dict[str, pd.Series] = {}
        raw_by_group: dict[str, pd.Series] = {}
        thr_by_group: dict[str, pd.Series] = {}
        flag_by_group: dict[str, pd.Series] = {}
        for name, g in groups.items():
            cols = g["columns"]
            scores_df, flags_df = self._score_one(
                panel_id=panel_id,
                label=f"group '{name}'",
                fit_true_df=fit_true_df[cols],
                fit_pred_df=fit_pred_df[cols],
                eval_true_df=eval_true_df[cols],
                eval_pred_df=eval_pred_df[cols],
                high_quantile=applied_q[name],
            )
            norm_by_group[name] = scores_df["anomaly_score_normalized"]
            raw_by_group[name] = scores_df["anomaly_score"]
            thr_by_group[name] = scores_df["anomaly_threshold"]
            flag_by_group[name] = flags_df["anomaly_flag"]

        index = eval_true_df.index
        flags_matrix = pd.DataFrame(flag_by_group, index=index).fillna(0).astype(int)
        norm_matrix = pd.DataFrame(norm_by_group, index=index)
        raw_matrix = pd.DataFrame(raw_by_group, index=index)
        thr_matrix = pd.DataFrame(thr_by_group, index=index)

        # System flag = OR across groups.
        system_flag = flags_matrix.max(axis=1).astype(int)

        # System score = the highest per-group normalized score, with the raw
        # score of that same (winning) group. argmax over -inf-filled values is
        # robust to all-NaN rows (a group that failed scoring).
        winner_pos = norm_matrix.fillna(-np.inf).to_numpy().argmax(axis=1)
        raw_values = raw_matrix.to_numpy()
        system_raw = raw_values[np.arange(len(winner_pos)), winner_pos]

        scores_df = pd.DataFrame(
            {
                "anomaly_score": system_raw,
                "anomaly_score_normalized": norm_matrix.max(axis=1),
            },
            index=index,
        )
        flags_out = pd.DataFrame({"anomaly_flag": system_flag}, index=index)

        # System raw flag threshold = the winning group's threshold at each ts.
        scores_df["anomaly_threshold"] = thr_matrix.to_numpy()[np.arange(len(winner_pos)), winner_pos]

        # Per-group detail columns (additive; downstream reads anomaly_score* / anomaly_flag).
        for name in groups:
            scores_df[f"group_{name}_normalized"] = norm_matrix[name]
            scores_df[f"group_{name}_raw"] = raw_matrix[name]
            scores_df[f"group_{name}_threshold"] = thr_matrix[name]
            flags_out[f"group_{name}_flag"] = flags_matrix[name]

        n_flagged = int(system_flag.sum())
        per_group_counts = {name: int(flags_matrix[name].sum()) for name in groups}
        self.logger.info(
            f"Panel {panel_id}: grouped detection flagged {n_flagged} timestamp(s) (per-group: {per_group_counts})"
        )
        return scores_df, flags_out

    def detect_panel(
        self, panel_id: str, df: pd.DataFrame, live: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """Detect anomalies for a single panel."""
        self.logger.info(f"Detecting anomalies for panel {panel_id}...")

        model_data = self.load_forecasting_model(panel_id)

        history_df, unseen_df = self._split_unseen_scoring_data(
            panel_id=panel_id,
            df=df,
            model_data=model_data,
        )

        hist_window = self.config["detect"]["hist_window"]
        if live:
            target_date = None
        else:
            target_date = self.config["detect"].get("target_date", None)

        if target_date:
            test_start_idx, test_end_idx = self._calculate_test_window_with_target_date(
                unseen_df, target_date, hist_window
            )
        else:
            test_start_idx, test_end_idx = self._calculate_test_window_default(unseen_df, hist_window)

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
        predictor = SpotforecastPredictor(self.config, self.logger)
        history_for_fit_pred = history_df if len(history_df) > 0 else None
        fit_pred_df = predictor.predict(
            model_data,
            scorer_fit_df,
            history_df=history_for_fit_pred,
        )
        eval_history_df = pd.concat([history_df, scorer_fit_df]) if len(history_df) > 0 else scorer_fit_df
        eval_pred_df = predictor.predict(model_data, scorer_eval_df, history_df=eval_history_df)

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

        # Multiply residuals by a multiply_residuals source's column so that
        # residual = column*(actual - predicted). This suppresses anomalies when
        # the column is small and amplifies them when it is large.
        weight_suffix = self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")
        mult_prefixes = multiplier_prefixes(self.config)
        multiplier_col = find_multiplier_column(scorer_fit_df.columns, mult_prefixes, weight_suffix)
        if multiplier_col is not None:
            fit_true_df, fit_pred_df, eval_true_df, eval_pred_df = self._apply_residual_multiplier(
                panel_id=panel_id,
                multiplier_col=multiplier_col,
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

        # Run scoring. When detect.groups defines groups for this panel, each
        # group is scored independently with a Bonferroni-corrected quantile and
        # their flags OR-ed; otherwise all channels are scored as one system.
        self.logger.info("Computing anomaly scores and detecting anomalies...")
        contributions_df: pd.DataFrame | None = None
        groups = self._resolve_groups(panel_id, list(eval_true_df.columns))
        if groups:
            scores_df, flags_df = self._score_grouped(
                panel_id=panel_id,
                groups=groups,
                fit_true_df=fit_true_df,
                fit_pred_df=fit_pred_df,
                eval_true_df=eval_true_df,
                eval_pred_df=eval_pred_df,
            )
        else:
            scores_df, flags_df = self._score_one(
                panel_id=panel_id,
                label="system",
                fit_true_df=fit_true_df,
                fit_pred_df=fit_pred_df,
                eval_true_df=eval_true_df,
                eval_pred_df=eval_pred_df,
                high_quantile=self.config["detect"]["high_quantile"],
            )

        if scores_df.index.tz is None:
            scores_df.index = scores_df.index.tz_localize("UTC")
            flags_df.index = flags_df.index.tz_localize("UTC")
            report_pred_df.index = report_pred_df.index.tz_localize("UTC")
            if contributions_df is not None:
                contributions_df.index = contributions_df.index.tz_localize("UTC")

        num_anomalies = int(flags_df["anomaly_flag"].sum())
        self.logger.info(f"Detected {num_anomalies} anomalies for panel {panel_id}")

        return scores_df, flags_df, report_pred_df, contributions_df

    def detect_all_panels(
        self, panel_data: dict[str, pd.DataFrame], live: bool = False
    ) -> dict[
        str,
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None],
    ]:
        results = {}
        for panel_id, df in panel_data.items():
            self.logger.info(f"Detecting anomalies for panel {panel_id}...")
            result = self.detect_panel(panel_id, df, live)
            results[panel_id] = result
            self.logger.info(f"Completed detection for panel {panel_id}")
        return results

    def run(
        self, panel_data: dict[str, pd.DataFrame], live: bool = False
    ) -> dict[
        str,
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None],
    ]:
        self.logger.info("Starting anomaly detection...")
        results = self.detect_all_panels(panel_data, live)
        self.logger.info("Anomaly detection completed successfully")
        return results
