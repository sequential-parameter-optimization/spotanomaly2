"""Column-role resolution for panel DataFrames.

Decides which columns of a processed panel are forecasting *targets* and which
are *exogenous* inputs. Used by both ``SpotforecastTrainer`` and
``SpotforecastTuner`` so the tuned hyperparameters match the topology the
trainer actually fits.
"""

import logging
from typing import Any

import pandas as pd


def _split_panel_columns(
    config: dict[str, Any],
    logger: logging.Logger,
    df: pd.DataFrame,
    weight_suffix: str,
) -> tuple[list[str], list[str]]:
    """Classify each column of ``df`` as either a forecasting target or an exog input.

    - **Exog set** = explicit ``exog_columns`` from config ∪ every
      ``exogenous_*``-prefixed column present in ``df`` — except when
      ``residual_weighting.enabled`` is True, in which case the auto-prefixed
      columns are dropped (residual weighting consumes them, so feeding them
      to the model as features would double-count). Order is preserved and
      duplicates removed; columns not present in ``df`` are silently skipped.
    - **Target set** = every remaining column except those ending in
      ``weight_suffix`` (imputation/weight columns are never targets) and
      except any leftover ``exogenous_*``-prefixed columns (they are exog
      features, never scored as targets — even when ``residual_weighting``
      keeps them out of the exog set).

    Returns ``(target_cols, exog_columns)`` in the order they appear in ``df``.
    Trainer and tuner both call this so the hyperparameter search and the
    final fit see the same model topology.
    """
    configured_exog_columns = config["train"].get("exog_columns", [])
    weight_residuals_enabled = config.get("residual_weighting", {}).get("enabled", False)
    if weight_residuals_enabled:
        logger.info("residual_weighting enabled: exogenous columns excluded from model features")
        prefixed_exog_columns: list[str] = []
    else:
        prefixed_exog_columns = [
            col for col in df.columns if col.startswith("exogenous_") and not col.endswith(weight_suffix)
        ]
    exog_columns = list(
        dict.fromkeys([col for col in [*configured_exog_columns, *prefixed_exog_columns] if col in df.columns])
    )
    target_cols = [
        col
        for col in df.columns
        if not col.endswith(weight_suffix) and col not in exog_columns and not col.startswith("exogenous_")
    ]
    return target_cols, exog_columns
