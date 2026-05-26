# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Download-stage orchestrator: runs every configured ExogenousFetcher.

Mirrors the responsibility of ``PrimaryDataFetcher`` for the *primary* data —
this class is to ``download()`` what ``DataProcessor`` is to ``process()``:
the single application-layer entry point that delegates per-source work to
plugin-supplied domain classes.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from spotanomaly2.domain.exogenous.registry import iter_configured_sources
from spotanomaly2.infrastructure import logging as infra_logging


class ExogenousDownloader:
    """Invokes ``fetch_and_cache`` on every configured ExogenousFetcher class."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or infra_logging.get_logger("ExogenousDownloader")

    def download_all(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fetch_status: Optional[dict] = None,
    ) -> None:
        for name, src_cfg, FetcherCls, _ in iter_configured_sources(self.config):
            if not FetcherCls.is_enabled(src_cfg):
                self._set_status(fetch_status, name, "disabled", None)
                continue
            try:
                FetcherCls(src_cfg, self.config, self.logger).fetch_and_cache(start, end)
                self._set_status(fetch_status, name, "ok", None)
            except Exception as exc:
                self.logger.warning(f"{name} fetch failed, continuing without it: {exc}")
                self._set_status(fetch_status, name, "error", str(exc))

    @staticmethod
    def _set_status(status: Optional[dict], name: str, state: str, error: Optional[str]) -> None:
        if status is None:
            return
        status[name] = {
            "status": state,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
