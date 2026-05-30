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

from typing import Any

import pandas as pd

from spotanomaly2.domain.exogenous.residual_multiplier import multiplier_prefixes

from .preprocessing import (
    _apply_known_anomaly_imputation,
    _build_strict_training_sample_mask,
    _detect_anomalies_via_ridge,
    _split_panel_columns,
)

# Sentinel returned by ``build_sample_mask`` when the strict-imputation mask
# leaves zero usable training samples — the caller should skip the channel.
SKIP_CHANNEL = "skip"


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
    observed-mask logic treats them as "not real" uniformly.
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
