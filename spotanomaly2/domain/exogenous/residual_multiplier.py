"""Per-source residual-multiplier helpers.

An exogenous source can opt in via ``multiply_residuals: true`` in its YAML
``config`` block. Such a source's columns do not become model features; instead
they multiply the detection residuals element-wise
(``residual = column * (actual - predicted)``), so anomalies count in proportion
to the column's magnitude (see ``AnomalyDetector._apply_residual_multiplier``).

Because the train and detect stages reload panels from disk, the column->source
mapping can't be carried in memory — it is reconstructed from config plus the
``exogenous_<source_name>_<measurement>`` naming convention that every joiner
follows. These helpers centralise that reconstruction so the splitter, detector,
and report generator agree on which columns are multipliers.
"""

from typing import Any, Iterable


def multiplier_prefixes(config: dict[str, Any]) -> list[str]:
    """Return the ``exogenous_<name>_`` column prefixes of multiplier sources.

    A source contributes its prefix iff its ``config.multiply_residuals`` is
    truthy. Sources without the flag (the default) produce ordinary exogenous
    features.
    """
    prefixes: list[str] = []
    for entry in config.get("exogenous", []) or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        src_cfg = entry.get("config", {}) or {}
        if name and src_cfg.get("multiply_residuals", False):
            prefixes.append(f"exogenous_{name}_")
    return prefixes


def is_multiplier_column(column: str, prefixes: Iterable[str]) -> bool:
    """True if *column* belongs to a residual-multiplier source."""
    return any(column.startswith(p) for p in prefixes)


def find_multiplier_column(
    columns: Iterable[str],
    prefixes: Iterable[str],
    weight_suffix: str = "__weight",
) -> str | None:
    """Return the first residual-multiplier column among *columns*, or None.

    The imputation observation-weight companions (``*<weight_suffix>``) are
    skipped — they are not measurement columns.
    """
    prefixes = list(prefixes)
    if not prefixes:
        return None
    for column in columns:
        if column.endswith(weight_suffix):
            continue
        if is_multiplier_column(column, prefixes):
            return column
    return None
