# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Process-stage orchestrator: runs every configured ExogenousJoiner.

Reads pre-fetched caches written by ``ExogenousDownloader`` and joins external
columns onto panel DataFrames before ``DataProcessor`` runs. No network I/O.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from spotanomaly2.domain.exogenous.registry import iter_configured_sources
from spotanomaly2.infrastructure import logging as infra_logging


class ExogenousJoinManager:
    """Invokes ``join_into_panels`` on every configured ExogenousJoiner class."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or infra_logging.get_logger("ExogenousJoiner")

    def join_all(
        self,
        panel_data: dict[str, pd.DataFrame],
        fetch_status: Optional[dict] = None,
    ) -> dict[str, pd.DataFrame]:
        for name, src_cfg, FetcherCls, JoinerCls in iter_configured_sources(self.config):
            if not FetcherCls.is_enabled(src_cfg):
                # Mirrors downloader's gating — if the source is off, neither stage runs.
                self._set_status(fetch_status, name, "disabled", None)
                continue
            try:
                joiner = JoinerCls(src_cfg, self.config, self.logger)
                panel_data = joiner.join_into_panels(panel_data)
                self._set_status(fetch_status, name, "ok", None)
            except Exception as exc:
                self.logger.warning(f"{name} join failed, continuing without it: {exc}")
                self._set_status(fetch_status, name, "error", str(exc))
        return panel_data

    @staticmethod
    def _set_status(status: Optional[dict], name: str, state: str, error: Optional[str]) -> None:
        if status is None:
            return
        status[name] = {
            "status": state,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
