# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Primary data fetcher backed by the ENTSO-E Transparency Platform.

The bundled reference implementation of the pluggable primary-fetcher contract
(selected via ``primary.fetcher`` in config), loading real electricity-load time
series from the public ENTSO-E API. It powers the self-contained
``config/entsoe_example.yaml`` example: any user with a free ENTSO-E security
token can run the full pipeline end-to-end, no proprietary credentials required.

The fetch is cache-free and always live — every ``run`` re-queries the configured
window up to "now" — so the example reflects current grid data on each invocation.
Network access is delegated to the ``entsoe-py`` client; we intentionally do not
reuse ``spotforecast2_safe.downloader.entsoe`` here because that helper persists
CSVs into its own data home, whereas the pipeline owns raw-data persistence via
``DataManager``.
"""

import os
import time
from typing import Any, cast

import pandas as pd
from dotenv import load_dotenv

from spotanomaly2.infrastructure import logging

_MAX_RETRIES = 5
_RETRY_BACKOFF_SECONDS = 5


class EntsoePrimaryFetcher:
    """Fetch electricity load from ENTSO-E and shape it into a single-channel panel.

    Config keys (under ``primary.config``):
        country_code: ENTSO-E bidding-zone / country code (default ``"DE"``).
            Also used as the panel id, so ``panels.panel_ids`` must contain it.
        channel_name: suffix for the target column ``channel_0_<channel_name>``
            (default ``"load"``).

    The fetch window comes from ``config['fetch']`` (``start_date`` required;
    ``end_date`` null → now). The ``ENTSOE_API_KEY`` token is read from the
    environment (``paths.credentials_file`` is loaded first, mirroring
    ``PrimaryDataFetcher``).
    """

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("EntsoePrimaryFetcher")
        primary_cfg = (config.get("primary") or {}).get("config") or {}
        self.country_code: str = primary_cfg.get("country_code", "DE")
        self.channel_name: str = primary_cfg.get("channel_name", "load")

    def _api_key(self) -> str:
        creds_path = self.config.get("paths", {}).get("credentials_file")
        if creds_path:
            load_dotenv(creds_path)
        api_key = os.getenv("ENTSOE_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing ENTSOE_API_KEY. Set it in your environment or in the file "
                "referenced by paths.credentials_file. Get a free security token at "
                "https://transparency.entsoe.eu (Account Settings → Web API)."
            )
        return api_key

    def _resolve_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        fetch_cfg = self.config.get("fetch", {})
        start_raw = fetch_cfg.get("start_date")
        if not start_raw:
            raise ValueError("config['fetch']['start_date'] is required for EntsoePrimaryFetcher")
        start = pd.Timestamp(start_raw)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        end_raw = fetch_cfg.get("end_date")
        if end_raw:
            end = pd.Timestamp(end_raw)
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
        else:
            end = pd.Timestamp.now(tz="UTC")
        # Config values parse to concrete Timestamps; cast away the stub's NaT union.
        return cast(pd.Timestamp, start), cast(pd.Timestamp, end)

    def _query_actual_load(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Query ENTSO-E and return the 'Actual Load' series (bounded retry)."""
        try:
            from entsoe import EntsoePandasClient
        except ImportError as e:
            raise ImportError(
                "The 'entsoe-py' library is required for EntsoePrimaryFetcher. Install it with: uv add entsoe-py"
            ) from e

        client = EntsoePandasClient(api_key=self._api_key())

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                self.logger.info(
                    f"Querying ENTSO-E load for {self.country_code} from {start} to {end} "
                    f"(attempt {attempt}/{_MAX_RETRIES})..."
                )
                df = client.query_load_and_forecast(country_code=self.country_code, start=start, end=end)
                return self._extract_actual_load(df)
            except Exception as e:  # noqa: BLE001 — bounded retry, re-raised below
                last_exc = e
                self.logger.warning(f"ENTSO-E query failed: {e}. Retrying in {_RETRY_BACKOFF_SECONDS}s...")
                time.sleep(_RETRY_BACKOFF_SECONDS)

        raise RuntimeError(
            f"Failed to fetch ENTSO-E load for {self.country_code} after {_MAX_RETRIES} attempts"
        ) from last_exc

    @staticmethod
    def _extract_actual_load(df: pd.DataFrame) -> pd.Series:
        """Pick the actual-load column from the ENTSO-E response and clean it."""
        if "Actual Load" in df.columns:
            series = df["Actual Load"]
        else:
            actual_cols = [c for c in df.columns if "actual" in str(c).lower()]
            series = df[actual_cols[0]] if actual_cols else df.iloc[:, 0]
        series = pd.to_numeric(series, errors="coerce").dropna()
        return series

    def run(self, incremental_only: bool = False, ignore_cache: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch the configured load window and return ``{country_code: panel_df}``.

        ``incremental_only`` and ``ignore_cache`` are accepted for parity with the
        primary-fetcher contract but are no-ops here: this source holds no local
        cache, so every call performs a full live fetch of the configured window.
        """
        start, end = self._resolve_window()
        series = self._query_actual_load(start, end)

        target_col = f"channel_0_{self.channel_name}"
        panel_df = pd.DataFrame({target_col: series})
        idx = pd.DatetimeIndex(panel_df.index)
        # ENTSO-E returns a tz-aware local index; normalise to UTC (localise if naive).
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        panel_df.index = idx
        panel_df.index.name = "timestamp"
        panel_df = panel_df.sort_index()

        self.logger.info(
            f"Fetched {len(panel_df)} ENTSO-E load rows for {self.country_code} "
            f"({panel_df.index.min()} → {panel_df.index.max()})"
        )
        return {self.country_code: panel_df}
