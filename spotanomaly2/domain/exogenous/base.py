# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Protocols for external data sources that enrich panel data.

Identity (the per-source ``name`` shown in ``fetch_status`` and logs) is carried
by the YAML entry, not by a class attribute — third-party classes only need to
implement the method shape.
"""

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ExogenousFetcher(Protocol):
    """Pulls one external data source and persists to its cache. No panel knowledge."""

    def __init__(
        self,
        source_config: dict[str, Any],
        parent_config: dict[str, Any],
        logger=None,
    ) -> None: ...

    @classmethod
    def is_enabled(cls, source_config: dict[str, Any]) -> bool:
        """Return True iff this source is configured to run (e.g. has credentials/coords)."""
        ...

    def fetch_and_cache(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Fetch ``[start, end]`` (extend internally if a baseline window is needed)
        and write the result to the source's cache. Idempotent; cache-aware."""
        ...


@runtime_checkable
class ExogenousJoiner(Protocol):
    """Reads the cache its paired Fetcher wrote and joins columns onto panels.

    Joiners must never trigger network I/O. If their cache is missing, they
    log a warning and return ``panel_data`` unchanged.
    """

    def __init__(
        self,
        source_config: dict[str, Any],
        parent_config: dict[str, Any],
        logger=None,
    ) -> None: ...

    def join_into_panels(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]: ...
