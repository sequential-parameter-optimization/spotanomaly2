"""Shared per-panel / per-channel preparation for the trainer and tuner.

``SpotforecastTrainer`` and ``SpotforecastTuner`` must prepare their data
*identically* — same target/exog split, same known-anomaly imputation, same
imputation-flag-aware sample weighting, same ``weight_func`` wiring — otherwise
the tuner optimises hyperparameters for a different objective than the trainer
ends up fitting. These helpers are the single source of truth for that shared
preparation so the two entry points can't drift apart.

They are deliberately plain functions taking ``config`` (and an optional
``logger``) explicitly rather than methods on a base class: the trainer and
tuner stay independent orchestrators that *call a service*, mirroring the rest
of the adapter's functional helper modules (``preprocessing``, ``factory`` …).
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from spotanomaly2.domain.exogenous.residual_multiplier import multiplier_prefixes
from spotanomaly2.domain.imputation_methods import impute_dataframe

from .prediction import _difference
from .preprocessing import (
    _apply_known_anomaly_imputation,
    _build_strict_training_sample_mask,
    _compute_observed_mask,
    _detect_anomalies_via_ridge,
    _ensure_freq,
    _split_panel_columns,
)

# Sentinel returned by ``build_sample_mask`` when the strict-imputation mask
# leaves zero usable training samples — the caller should skip the channel.
SKIP_CHANNEL = "skip"


@dataclass(frozen=True)
class TrainKnobs:
    """Train-side knobs the trainer fits with and the tuner must mirror.

    Resolved once from ``config['train']`` (+ the imputation weight suffix) so
    the trainer and tuner can't read different defaults for the same setting.
    ``n_lags`` is the configured lag spec (int or list); ``n_lags_max`` is the
    integer upper bound used for sufficiency checks and the strict-sample mask
    width — the tuner needs this because its search may pick any lag up to the
    largest in the search space.
    """

    n_lags: int | list[int]
    n_lags_max: int
    random_seed: int
    diff_order: int
    weight_suffix: str


def resolve_train_settings(config: dict[str, Any]) -> TrainKnobs:
    """Resolve the shared train-side knobs (lags, seed, Δy order, weight suffix).

    Single source of truth for the settings the trainer fits with and the tuner
    optimises against — see :class:`TrainKnobs`.
    """
    train_cfg = config.get("train", {})
    n_lags = train_cfg.get("lags", 24)
    if isinstance(n_lags, (list, tuple)) and n_lags:
        n_lags_max = int(max(n_lags))
    else:
        try:
            n_lags_max = int(n_lags)
        except (TypeError, ValueError):
            n_lags_max = 24
    return TrainKnobs(
        n_lags=n_lags,
        n_lags_max=n_lags_max,
        random_seed=train_cfg.get("random_seed", 42),
        # Δy order — must match what train_panel fits, otherwise the tuner's
        # winning hyperparameters aren't optimal for the production estimator.
        diff_order=int(train_cfg.get("differentiation", 1)),
        weight_suffix=get_weight_suffix(config),
    )


@dataclass
class ChannelData:
    """One channel's prepared fit inputs, identical for trainer and tuner.

    ``y_fit`` is what the forecaster fits against (Δy with leading NaNs dropped
    when ``diff_order > 0``, otherwise the raw series). ``y_raw`` is the
    pre-difference series — the trainer needs it to build its one-step-ahead
    test window and the tuner needs it to rescale R² back to raw space.
    ``exog_fit`` and ``sample_mask`` are imputed/aligned to ``y_fit.index``.
    """

    y_fit: pd.Series
    y_raw: pd.Series
    exog_fit: pd.DataFrame | None
    sample_mask: pd.Series | None


def get_weight_suffix(config: dict[str, Any]) -> str:
    """Return the imputation weight-column suffix (default ``"__weight"``)."""
    return config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")


def prepare_panel(
    config: dict[str, Any],
    df: pd.DataFrame,
    weight_suffix: str,
    logger=None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Split a panel into target/exog columns and apply known-anomaly masking.

    Returns ``(panel_data, target_cols, exog_columns)``. ``exogenous_*`` and
    ``multiply_residuals`` columns become exog features, never tuning/training
    targets. Known-anomaly windows are blanked and re-imputed with the same
    method the process stage used, with ``__weight`` merged so downstream
    observed-mask logic treats them as "not real" uniformly. The DataFrame's
    ``index.freq`` is set via ``_ensure_freq`` so the returned panel can be
    used directly by spotforecast2 (which requires a frequency-aware index).
    """
    configured_exog_columns = config["train"].get("exog_columns", [])
    mult_prefixes = multiplier_prefixes(config)
    if mult_prefixes and logger is not None:
        logger.info(f"multiply_residuals sources: columns {mult_prefixes} excluded from model features")
    target_cols, exog_columns = _split_panel_columns(df, configured_exog_columns, weight_suffix, mult_prefixes)

    known_anomalies = config.get("known_anomalies", [])
    known_anomaly_buffer = config["train"].get("known_anomaly_buffer")
    if known_anomalies and known_anomaly_buffer:
        imp_cfg = config.get("process", {}).get("imputation", {})
        df = _apply_known_anomaly_imputation(
            df,
            known_anomalies,
            known_anomaly_buffer,
            target_cols=target_cols,
            weight_suffix=weight_suffix,
            imputation_method=imp_cfg.get("method", "linear_interpolation"),
            imputation_params=imp_cfg.get("params", {}),
        )

    fallback_freq = config.get("process", {}).get("resample", {}).get("freq", "5min")
    df = _ensure_freq(df, fallback_freq)

    return df, target_cols, exog_columns


def build_sample_mask(
    config: dict[str, Any],
    observed_mask: pd.Series,
    y_train: pd.Series,
    target_col: str,
    n_lags: int,
    logger=None,
) -> pd.Series | str | None:
    """Combine the strict-imputation mask with an optional auto-clean anomaly mask.

    Returns:
        - ``None`` if no weighting is needed,
        - :data:`SKIP_CHANNEL` if the strict mask leaves zero usable samples,
        - a ``pd.Series`` of bool weights otherwise.
    """
    sample_mask: pd.Series | None = None
    train_cfg = config.get("train", {})

    if train_cfg.get("exclude_imputed_training_samples", False):
        sample_mask = _build_strict_training_sample_mask(observed_mask=observed_mask, n_lags=n_lags)
        potential = max(len(y_train) - n_lags, 0)
        kept = int(sample_mask.iloc[n_lags:].sum())
        if logger is not None:
            logger.info(
                f"    {target_col}: excluding {potential - kept} training sample(s) "
                f"that contain imputed target/lag values"
            )
        if kept == 0:
            return SKIP_CHANNEL

    if train_cfg.get("auto_clean_anomalies", False):
        anomaly_mask = _detect_anomalies_via_ridge(
            y_train,
            n_lags=n_lags,
            threshold_scale=train_cfg.get("auto_clean_threshold", 4.0),
            buffer=train_cfg.get("auto_clean_buffer", 3),
        )
        n_flagged = int(anomaly_mask.sum())
        if n_flagged > 0:
            if logger is not None:
                logger.info(f"    {target_col}: auto-cleaning flagged {n_flagged} suspected anomaly points")
            if sample_mask is not None:
                sample_mask = sample_mask & ~anomaly_mask
            else:
                sample_mask = ~anomaly_mask

    return sample_mask


def impute_exog(config: dict[str, Any], exog_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill exog NaN cells using the config's imputation method.

    Uses ``process.imputation.{method,params}`` — the same single source of
    truth the process stage and known-anomaly masking use — rather than a
    hard-coded interpolation, so the exog features the trainer, tuner, and
    predictor see are filled identically. Returns ``exog_df`` unchanged when it
    has no missing cells.
    """
    if not exog_df.isna().any().any():
        return exog_df
    imp_cfg = config.get("process", {}).get("imputation", {})
    return impute_dataframe(
        exog_df,
        method=imp_cfg.get("method", "linear_interpolation"),
        columns=columns,
        **imp_cfg.get("params", {}),
    )


def attach_weight_func(forecaster, sample_mask: pd.Series | None) -> None:
    """Attach a ``weight_func`` that zeros out masked (imputed/anomalous) rows.

    No-op when ``sample_mask`` is None. The closure re-aligns the mask to
    whatever index spotforecast2 calls it with, so it works for both the raw
    and the differenced training index.
    """
    if sample_mask is None:
        return

    def _weight_func(index, mask=sample_mask):
        return mask.reindex(index).fillna(False).astype(float).to_numpy()

    forecaster.weight_func = _weight_func


def prepare_channel(
    config: dict[str, Any],
    df: pd.DataFrame,
    target_col: str,
    exog_columns: list[str],
    weight_suffix: str,
    n_lags_for_mask: int,
    diff_order: int,
    logger=None,
) -> ChannelData | None:
    """Build one channel's fit inputs (target, exog, sample mask, Δy) for both paths.

    The single source of truth for per-channel setup so the trainer and tuner
    can't diverge on target extraction, sufficiency thresholds, imputation-flag
    masking, exog imputation, or differentiation. Returns ``None`` when the
    channel must be skipped — too few rows, or the strict-imputation mask leaves
    zero usable samples.

    ``n_lags_for_mask`` is the lag count used for the sufficiency check and the
    strict-sample mask width: the trainer passes the channel's resolved (max)
    lag, the tuner passes the search-space upper bound.
    """
    y = df[target_col].copy()
    y.name = target_col

    observed_mask = _compute_observed_mask(df, target_col, weight_suffix)
    if len(y) < n_lags_for_mask + 10:
        if logger is not None:
            logger.warning(f"  Skipping {target_col}: insufficient data ({len(y)} rows)")
        return None

    sample_mask = build_sample_mask(config, observed_mask, y, target_col, n_lags_for_mask, logger)
    if sample_mask is SKIP_CHANNEL:
        if logger is not None:
            logger.warning(f"  Skipping {target_col}: no fully observed training samples left")
        return None

    exog_fit: pd.DataFrame | None = None
    if exog_columns:
        exog_fit = df[exog_columns].loc[y.index]
        exog_fit = impute_exog(config, exog_fit, exog_columns)

    # Fit on Δy when differentiation is on (matches train_panel). The raw series
    # is kept for the trainer's test window and the tuner's raw-space R².
    y_raw = y.copy()
    y_fit = y
    if diff_order > 0:
        y_fit = _difference(y, diff_order).dropna()
        if exog_fit is not None:
            exog_fit = exog_fit.loc[y_fit.index]
        if isinstance(sample_mask, pd.Series):
            sample_mask = sample_mask.reindex(y_fit.index).fillna(False)

    return ChannelData(
        y_fit=y_fit,
        y_raw=y_raw,
        exog_fit=exog_fit,
        sample_mask=sample_mask if isinstance(sample_mask, pd.Series) else None,
    )
