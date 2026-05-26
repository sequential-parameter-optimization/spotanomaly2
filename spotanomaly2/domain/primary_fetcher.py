"""Data fetching service for downloading data from API."""

from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from spotanomaly2.infrastructure import logging, storage
from spotanomaly2.infrastructure.auth import OAuthSession


class _FetchRange(NamedTuple):
    """One time window the fetcher needs to download. ``reason`` is ``full``/``before``/``after``."""

    start: str
    end: str | None
    reason: str


class PrimaryDataFetcher:
    """Service for fetching data from API and saving as Parquet."""

    def __init__(self, config: dict[str, Any], logger=None):
        """Initialize DataFetcher with configuration.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.get_logger("PrimaryFetcher")
        self.oauth_session = None

    def _init_oauth_session(self) -> OAuthSession:
        """Initialize OAuth session with credentials from config."""
        import os

        from dotenv import load_dotenv

        # Load credentials
        creds_path = self.config["paths"]["credentials_file"]
        load_dotenv(creds_path)

        api_key = os.getenv("PRIMARY_API_KEY")
        api_secret = os.getenv("PRIMARY_API_SECRET")
        tech_user = os.getenv("PRIMARY_TECH_USERNAME")
        tech_password = os.getenv("PRIMARY_TECH_PASSWORD")

        if not all([api_key, api_secret, tech_user, tech_password]):
            raise ValueError("Missing required credentials in environment file")

        return OAuthSession(
            api_key=api_key,
            api_secret=api_secret,
            tech_user=tech_user,
            tech_password=tech_password,
            token_request_url=self.config["fetch"]["token_request_url"],
            api_request_url=self.config["fetch"]["api_request_url"],
            logger=self.logger,
        )

    def fetch_panel_data(
        self,
        panel_id: str,
        start_date_override: str | None = None,
        end_date_override: str | None = None,
    ) -> pd.DataFrame:
        """Fetch data for a single panel and convert to DataFrame.

        Args:
            panel_id: Panel identifier
            start_date_override: Optional start date to override config (for sub-range fetch)
            end_date_override: Optional end date to override config

        Returns:
            DataFrame with multi-level columns (channel, key) and timestamp index
        """
        session = self._ensure_oauth_session()
        start_date, end_date = self._resolve_fetch_window(panel_id, start_date_override, end_date_override)
        values = self._fetch_raw_values(session, panel_id, start_date, end_date)
        series_list = self._build_series_list(panel_id, values)
        return self._assemble_dataframe(panel_id, series_list)

    def _ensure_oauth_session(self) -> OAuthSession:
        """Lazily initialize the OAuth session on first use; return the live session."""
        if self.oauth_session is None:
            self.oauth_session = self._init_oauth_session()
        return self.oauth_session

    def _resolve_fetch_window(
        self,
        panel_id: str,
        start_date_override: str | None,
        end_date_override: str | None,
    ) -> tuple[str, str | None]:
        """Return the effective (start, end) for the fetch, preferring overrides over config."""
        start_date = start_date_override if start_date_override else self.config["fetch"]["start_date"]
        end_date = end_date_override if end_date_override is not None else self.config["fetch"].get("end_date")

        if start_date_override:
            self.logger.info(f"Panel {panel_id}: Fetching sub-range {start_date} → {end_date or 'now'}")
        else:
            self.logger.info(f"Panel {panel_id}: Using full fetch from {start_date} to {end_date or 'now'}")
        return start_date, end_date

    def _fetch_raw_values(
        self,
        session: OAuthSession,
        panel_id: str,
        start_date: str,
        end_date: str | None,
    ) -> dict[str, dict[str, list]]:
        """Call the API for every (channel, key) pair on this panel; return the raw value lists."""
        channels = self.config["panels"]["channel_ids"]
        channel_keys = self.config["panels"]["channel_keys"]
        ids = self.config["panels"]["instrumentation_ids"]

        values: dict[str, dict[str, list]] = {}
        for channel in channels:
            values[channel] = {}
            for key in channel_keys[channel]:
                path = f"instrumentations/{ids[panel_id][channel]}/values/{key}"
                self.logger.info(f"Fetching panel {panel_id}, {path}...")
                values[channel][key] = session.fetch_all_pages(
                    path, base_params={"from": start_date, "to": end_date}, per_page=1000
                )
        return values

    def _build_series_list(
        self,
        panel_id: str,
        values: dict[str, dict[str, list]],
    ) -> list[pd.Series]:
        """Turn raw API value lists into named, timestamp-indexed Series; skip empty channels."""
        channels = self.config["panels"]["channel_ids"]
        channel_keys = self.config["panels"]["channel_keys"]

        series_list: list[pd.Series] = []
        for channel in channels:
            for key in channel_keys[channel]:
                data = values[channel][key]
                if not data:
                    self.logger.warning(f"No data for panel {panel_id}, channel {channel}, key {key}")
                    continue
                series = pd.Series({x["timestamp"]: x["value"] for x in data})
                series.index = pd.to_datetime(series.index, utc=True)
                series.name = f"channel_{channel}_{key}"
                series_list.append(series)
        return series_list

    def _assemble_dataframe(self, panel_id: str, series_list: list[pd.Series]) -> pd.DataFrame:
        """Concat per-channel series; return an empty timestamp-indexed frame if none."""
        if not series_list:
            self.logger.critical(f"No data fetched for panel {panel_id} in requested range")
            return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))

        df = pd.concat(series_list, axis=1)
        df.index.name = "timestamp"
        self.logger.info(f"Fetched {len(df)} rows for panel {panel_id}")
        return df

    def run(self, incremental_only: bool = False) -> dict[str, pd.DataFrame]:
        """Run the complete fetch pipeline with incremental loading support.

        Checks if existing data covers the requested time range from config.
        Fetches missing data at the beginning or end of the range as needed.

        Args:
            incremental_only: If True, returns only newly fetched data without merging with existing.
                            If False (default), merges new data with existing data.

        Returns:
            Dictionary mapping panel_id to DataFrame (only new data if incremental_only=True,
                                                     merged with existing data otherwise)
        """
        self.logger.info("Starting data fetch...")

        raw_dir = Path(self.config["paths"]["raw_dir"])
        panels = self.config["panels"]["panel_ids"]
        requested_start = self.config["fetch"]["start_date"]
        requested_end = self.config["fetch"].get("end_date")  # None means "until now"

        panel_data = {}
        for panel_id in panels:
            panel_data[panel_id] = self._run_panel(panel_id, raw_dir, requested_start, requested_end, incremental_only)

        self.logger.info("Data fetch completed successfully")
        return panel_data

    def _run_panel(
        self,
        panel_id: str,
        raw_dir: Path,
        requested_start: str,
        requested_end: str | None,
        incremental_only: bool,
    ) -> pd.DataFrame:
        """Fetch (incrementally if possible) and combine the data for a single panel."""
        existing_df = self._load_existing(raw_dir, panel_id)
        fetch_ranges = self._plan_fetch_ranges(panel_id, existing_df, requested_start, requested_end)
        fetched_dfs = self._execute_fetch_ranges(panel_id, fetch_ranges)
        return self._combine_panel_data(panel_id, existing_df, fetched_dfs, incremental_only)

    def _load_existing(self, raw_dir: Path, panel_id: str) -> pd.DataFrame | None:
        """Load the most recent on-disk panel data, or None if it doesn't exist."""
        try:
            existing_df = storage.load_panel_parquet_versioned(raw_dir, panel_id)
        except FileNotFoundError:
            self.logger.info(f"No existing data found for panel {panel_id}")
            return None
        self.logger.info(f"Loaded {len(existing_df)} existing rows for panel {panel_id}")
        self.logger.info(f"Existing data range: {existing_df.index.min()} to {existing_df.index.max()}")
        return existing_df

    def _plan_fetch_ranges(
        self,
        panel_id: str,
        existing_df: pd.DataFrame | None,
        requested_start: str,
        requested_end: str | None,
    ) -> list[_FetchRange]:
        """Decide which time windows still need fetching given the existing data."""
        if existing_df is None:
            self.logger.info(f"Panel {panel_id}: Fetching entire range from {requested_start}")
            return [_FetchRange(start=requested_start, end=requested_end, reason="full")]

        existing_start = existing_df.index.min()
        existing_end = existing_df.index.max()
        ranges: list[_FetchRange] = []

        before_range = self._plan_before_range(panel_id, existing_start, requested_start)
        if before_range is not None:
            ranges.append(before_range)

        after_range = self._plan_after_range(panel_id, existing_end, requested_end)
        if after_range is not None:
            ranges.append(after_range)

        return ranges

    def _plan_before_range(
        self,
        panel_id: str,
        existing_start: pd.Timestamp,
        requested_start: str,
    ) -> _FetchRange | None:
        """Plan a fetch for the gap before existing data, skipping sub-2-minute gaps."""
        requested_start_ts = pd.to_datetime(requested_start, utc=True)
        if requested_start_ts >= existing_start:
            return None

        gap_duration = existing_start - requested_start_ts
        if gap_duration <= pd.Timedelta(minutes=2):
            self.logger.info(
                f"Panel {panel_id}: Skipping insignificant gap before existing data "
                f"({gap_duration.total_seconds():.1f} seconds)"
            )
            return None

        gap_end = (existing_start - pd.Timedelta(microseconds=1)).isoformat()
        self.logger.info(
            f"Panel {panel_id}: Gap detected before existing data ({gap_duration.total_seconds() / 3600:.1f} hours)"
        )
        return _FetchRange(start=requested_start, end=gap_end, reason="before")

    def _plan_after_range(
        self,
        panel_id: str,
        existing_end: pd.Timestamp,
        requested_end: str | None,
    ) -> _FetchRange | None:
        """Plan a fetch for the gap after existing data; always fetches if requested_end is open."""
        fetch_start = existing_end.isoformat()
        if requested_end is None:
            self.logger.info(f"Panel {panel_id}: Fetching new data from {fetch_start}")
            return _FetchRange(start=fetch_start, end=None, reason="after")

        requested_end_ts = pd.to_datetime(requested_end, utc=True)
        if requested_end_ts <= existing_end:
            return None

        self.logger.info(f"Panel {panel_id}: Gap detected after existing data")
        return _FetchRange(start=fetch_start, end=requested_end, reason="after")

    def _execute_fetch_ranges(self, panel_id: str, fetch_ranges: list[_FetchRange]) -> list[pd.DataFrame]:
        """Fetch each planned range; return only the non-empty results."""
        fetched_dfs: list[pd.DataFrame] = []
        for fetch_range in fetch_ranges:
            self.logger.info(
                f"Panel {panel_id}: Fetching {fetch_range.reason} range "
                f"from {fetch_range.start} to {fetch_range.end or 'now'}"
            )
            new_df = self.fetch_panel_data(
                panel_id,
                start_date_override=fetch_range.start,
                end_date_override=fetch_range.end,
            )
            if len(new_df) > 0:
                fetched_dfs.append(new_df)
                self.logger.info(f"Panel {panel_id}: Fetched {len(new_df)} rows for {fetch_range.reason} range")
            else:
                self.logger.info(f"Panel {panel_id}: No new data in {fetch_range.reason} range")
        return fetched_dfs

    def _combine_panel_data(
        self,
        panel_id: str,
        existing_df: pd.DataFrame | None,
        fetched_dfs: list[pd.DataFrame],
        incremental_only: bool,
    ) -> pd.DataFrame:
        """Concat existing (unless incremental_only) and fetched frames; dedup + sort."""
        parts = list(fetched_dfs)
        if existing_df is not None and not incremental_only:
            parts.insert(0, existing_df)

        if not parts:
            raise ValueError(f"No data available for panel {panel_id} in requested range")
        if len(parts) == 1:
            return parts[0]

        combined = pd.concat(parts)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        if incremental_only:
            self.logger.info(f"Panel {panel_id}: Fetched {len(combined)} new rows (incremental mode)")
        else:
            self.logger.info(f"Panel {panel_id}: Combined data has {len(combined)} total rows")
        return combined
