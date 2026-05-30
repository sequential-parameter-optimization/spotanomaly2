# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve the pipeline's primary data fetcher from config.

The primary source is pluggable in the same spirit as the exogenous sources
(see ``domain/exogenous/registry.py``): a config names a fetcher class via a
``"module:ClassName"`` spec under ``primary.fetcher``. spotanomaly2 ships no
proprietary primary source — the public
:class:`spotanomaly2.examples.entsoe_fetcher.EntsoePrimaryFetcher` is the
bundled reference implementation, and deployments point ``primary.fetcher`` at
their own (possibly third-party) fetcher.

A primary fetcher must satisfy :class:`PrimaryFetcher`: constructed with the
full ``config`` (it reads its own settings from it) and exposing
``run(incremental_only=False, ignore_cache=False)`` returning
``{panel_id: DataFrame}``.
"""

import importlib
from typing import Any, Protocol, cast, runtime_checkable

import pandas as pd

from spotanomaly2.infrastructure import logging


@runtime_checkable
class PrimaryFetcher(Protocol):
    """Interface a primary data fetcher must satisfy to be pluggable into ``Pipeline``."""

    def __init__(self, config: dict[str, Any], logger=None) -> None: ...

    def run(self, incremental_only: bool = False, ignore_cache: bool = False) -> dict[str, pd.DataFrame]: ...


def _load_class(spec: str) -> type:
    """Resolve a ``'module.path:ClassName'`` spec to the actual class."""
    if ":" not in spec:
        raise ValueError(f"Primary fetcher spec must be 'module:ClassName', got: {spec!r}")
    module_name, class_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def resolve_primary_fetcher(config: dict[str, Any], logger=None) -> PrimaryFetcher:
    """Return the primary fetcher named by ``config['primary']['fetcher']``.

    Raises:
        ValueError: when ``primary.fetcher`` is not configured. There is no
            implicit default — set it to a ``"module:ClassName"`` spec (see
            ``spotanomaly2.examples.entsoe_fetcher:EntsoePrimaryFetcher`` for a
            runnable example, and ``config/entsoe_example.yaml`` for a full config).
    """
    logger = logger or logging.get_logger("PrimaryRegistry")
    spec = (config.get("primary") or {}).get("fetcher")
    if not spec:
        raise ValueError(
            "No primary data fetcher configured. Set `primary.fetcher` to a "
            "'module:ClassName' spec implementing the primary-fetcher contract. "
            "Example: spotanomaly2.examples.entsoe_fetcher:EntsoePrimaryFetcher "
            "(see config/entsoe_example.yaml)."
        )
    fetcher_cls = _load_class(spec)
    logger.info(f"Using primary fetcher: {spec}")
    return cast(PrimaryFetcher, fetcher_cls(config, logger))
