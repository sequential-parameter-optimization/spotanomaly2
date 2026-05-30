"""Batch one-step-ahead prediction against a trained spotforecast2 model.

Separated from ``SpotforecastTrainer`` because predicting and training are
distinct lifecycle phases with different inputs (trainer takes raw panel data
and writes a model artifact; predictor takes a loaded model artifact + new
data and returns predictions). The ``predict`` method used to live on the
trainer for historical reasons — the class name no longer matched its
responsibility.
"""

from typing import Any

import numpy as np
import pandas as pd

from spotanomaly2.infrastructure import logging

from .channel_prep import impute_exog
from .prediction import _predict_one_step_integrated
from .preprocessing import _ensure_freq


class SpotforecastPredictor:
    """Runs batch one-step-ahead inference against a model produced by ``SpotforecastTrainer``."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("SpotforecastPredictor")

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

        df = _ensure_freq(df)
        if history_df is not None:
            history_df = _ensure_freq(history_df)

        predictions: dict[str, np.ndarray] = {}
        for target_col in target_cols:
            predictions[target_col] = self._predict_channel(
                forecasters.get(target_col),
                target_col,
                df,
                history_df,
                exog_columns,
                diff_order,
            )
        return pd.DataFrame(predictions, index=df.index)

    def _predict_channel(
        self,
        forecaster,
        target_col: str,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None,
        exog_columns: list[str],
        diff_order: int,
    ) -> np.ndarray:
        """Predict one channel; return an all-NaN array if the forecaster is missing or the call fails."""
        if forecaster is None:
            return np.full(len(df), np.nan)

        full_y = self._stitch_observed_target(target_col, df, history_df)
        if len(full_y) == 0:
            return np.full(len(df), np.nan)

        exog_full = self._stitch_exog(exog_columns, full_y, df, history_df)

        try:
            return _predict_one_step_integrated(forecaster, full_y, exog_full, df.index, diff_order)
        except Exception as e:
            self.logger.warning(f"Prediction failed for {target_col}: {e}. Using NaN.")
            return np.full(len(df), np.nan)

    @staticmethod
    def _stitch_observed_target(
        target_col: str,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None,
    ) -> pd.Series:
        """Concat history + df target into a single Series for the lag matrix.

        Trusts preprocessing for imputation — process-stage `ImputationStep`
        leaves the target gap-free, so no re-interpolation is needed here.
        """
        if history_df is not None:
            full_y = pd.concat([history_df[target_col], df[target_col]])
        else:
            full_y = df[target_col].copy()
        full_y.name = target_col
        return _ensure_freq(full_y)

    def _stitch_exog(
        self,
        exog_columns: list[str],
        full_y: pd.Series,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """Concat history+df exog columns aligned to ``full_y.index``; impute NaN cells.

        Gaps are filled with the config's imputation method (matching what the
        trainer/tuner do for exog) — with ``transformer_exog`` active on linear
        models, NaN cells would otherwise propagate through
        StandardScaler.transform and crash estimator.predict.
        """
        if not exog_columns:
            return None
        cols_present = [c for c in exog_columns if c in df.columns]
        if not cols_present:
            return None
        if history_df is not None:
            exog_full = pd.concat([history_df[cols_present], df[cols_present]])
        else:
            exog_full = df[cols_present].copy()
        exog_full = exog_full.loc[full_y.index]
        return impute_exog(self.config, exog_full, cols_present)
