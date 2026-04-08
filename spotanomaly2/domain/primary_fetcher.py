"""Data fetching service for downloading data from API."""

from pathlib import Path
from typing import Any

import pandas as pd

from spotanomaly2.infrastructure import logging
from spotanomaly2.infrastructure.auth import OAuthSession


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
        self, panel_id: str, start_date_override: str = None, end_date_override: str = None
    ) -> pd.DataFrame:
        """Fetch data for a single panel and convert to DataFrame.

        Args:
            panel_id: Panel identifier
            start_date_override: Optional start date to override config (for incremental fetch)
            end_date_override: Optional end date to override config

        Returns:
            DataFrame with multi-level columns (channel, key) and timestamp index
        """
        if self.oauth_session is None:
            self.oauth_session = self._init_oauth_session()

        channels = self.config["panels"]["channel_ids"]
        channel_keys = self.config["panels"]["channel_keys"]
        ids = self.config["panels"]["instrumentation_ids"]

        # Use override if provided, otherwise use config
        start_date = start_date_override if start_date_override else self.config["fetch"]["start_date"]
        end_date = end_date_override if end_date_override is not None else self.config["fetch"].get("end_date")

        if start_date_override:
            self.logger.info(f"Panel {panel_id}: Using incremental fetch from {start_date} to {end_date or 'now'}")
        else:
            self.logger.info(f"Panel {panel_id}: Using full fetch from {start_date} to {end_date or 'now'}")

        # Fetch raw values
        values = {}
        for channel in channels:
            values[channel] = {}
            for key in channel_keys[channel]:
                path = f"instrumentations/{ids[panel_id][channel]}/values/{key}"
                self.logger.info(f"Fetching panel {panel_id}, {path}...")

                values[channel][key] = self.oauth_session.fetch_all_pages(
                    path, base_params={"from": start_date, "to": end_date}, per_page=1000
                )

        # Convert to DataFrame
        series_list = []
        for channel in channels:
            for key in channel_keys[channel]:
                # Extract timestamp-value pairs
                data = values[channel][key]
                if not data:
                    self.logger.warning(f"No data for panel {panel_id}, channel {channel}, key {key}")
                    continue

                series = pd.Series({x["timestamp"]: x["value"] for x in data})
                series.index = pd.to_datetime(series.index, utc=True)
                series.name = f"channel_{channel}_{key}"
                series_list.append(series)

        # Combine all series into single DataFrame
        if not series_list:
            self.logger.warning(f"No data fetched for panel {panel_id} in requested range")
            # Return empty DataFrame with proper structure
            return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))

        df = pd.concat(series_list, axis=1)
        df.index.name = "timestamp"

        self.logger.info(f"Fetched {len(df)} rows for panel {panel_id}")
        return df

    def fetch_all_panels(self, start_dates: dict[str, str] = None) -> dict[str, pd.DataFrame]:
        """Fetch data for all panels.

        Args:
            start_dates: Optional dict mapping panel_id to start_date for incremental fetch

        Returns:
            Dictionary mapping panel_id to DataFrame
        """
        panels = self.config["panels"]["panel_ids"]
        panel_data = {}

        self.logger.info(f"Fetching data for {len(panels)} panels...")
        for panel_id in panels:
            start_date_override = start_dates.get(panel_id) if start_dates else None
            panel_data[panel_id] = self.fetch_panel_data(panel_id, start_date_override)

        return panel_data

    def run(self, start_dates: dict[str, str] = None, incremental_only: bool = False) -> dict[str, pd.DataFrame]:
        """Run the complete fetch pipeline with incremental loading support.

        Checks if existing data covers the requested time range from config.
        Fetches missing data at the beginning or end of the range as needed.

        Args:
            start_dates: Optional dict mapping panel_id to start_date for incremental fetch.
                        If None, will check for existing data and compare with config range.
            incremental_only: If True, returns only newly fetched data without merging with existing.
                            If False (default), merges new data with existing data.

        Returns:
            Dictionary mapping panel_id to DataFrame (only new data if incremental_only=True,
                                                     merged with existing data otherwise)
        """
        self.logger.info("Starting data fetch...")

        import pandas as pd

        from spotanomaly2.infrastructure import storage

        raw_dir = Path(self.config["paths"]["raw_dir"])
        panels = self.config["panels"]["panel_ids"]

        # Get requested time range from config
        requested_start = self.config["fetch"]["start_date"]
        requested_end = self.config["fetch"].get("end_date")  # None means "until now"

        # Parse requested start as timestamp for comparison
        requested_start_ts = pd.to_datetime(requested_start, utc=True)

        panel_data = {}

        for panel_id in panels:
            # Try to load existing data
            existing_df = None
            try:
                existing_df = storage.load_panel_parquet_versioned(raw_dir, panel_id)
                self.logger.info(f"Loaded {len(existing_df)} existing rows for panel {panel_id}")
                existing_start = existing_df.index.min()
                existing_end = existing_df.index.max()
                self.logger.info(f"Existing data range: {existing_start} to {existing_end}")
            except FileNotFoundError:
                self.logger.info(f"No existing data found for panel {panel_id}")
                existing_df = None

            # Determine what data needs to be fetched
            fetch_ranges = []

            if existing_df is None:
                # No existing data - fetch entire requested range
                self.logger.info(f"Panel {panel_id}: Fetching entire range from {requested_start}")
                fetch_ranges.append({"start": requested_start, "end": requested_end, "reason": "full"})
            else:
                # Check if we need data BEFORE existing data
                # Only fetch if gap is significant (more than 2 minutes)
                if requested_start_ts < existing_start:
                    gap_duration = existing_start - requested_start_ts
                    if gap_duration > pd.Timedelta(minutes=2):
                        gap_end = (existing_start - pd.Timedelta(microseconds=1)).isoformat()
                        self.logger.info(
                            f"Panel {panel_id}: Gap detected before existing data "
                            f"({gap_duration.total_seconds() / 3600:.1f} hours)"
                        )
                        fetch_ranges.append({"start": requested_start, "end": gap_end, "reason": "before"})
                    else:
                        self.logger.info(
                            f"Panel {panel_id}: Skipping insignificant gap before existing data "
                            f"({gap_duration.total_seconds():.1f} seconds)"
                        )

                # Check if we need data AFTER existing data
                # If requested_end is None, always check for new data
                if requested_end is None:
                    # Fetch from last timestamp until now
                    fetch_start = existing_end.isoformat()
                    self.logger.info(f"Panel {panel_id}: Fetching new data from {fetch_start}")
                    fetch_ranges.append({"start": fetch_start, "end": None, "reason": "after"})
                else:
                    # Check if requested end is after existing end
                    requested_end_ts = pd.to_datetime(requested_end, utc=True)
                    if requested_end_ts > existing_end:
                        fetch_start = existing_end.isoformat()
                        self.logger.info(f"Panel {panel_id}: Gap detected after existing data")
                        fetch_ranges.append({"start": fetch_start, "end": requested_end, "reason": "after"})

            # Fetch data for all ranges
            fetched_dfs = []
            if existing_df is not None and not incremental_only:
                # Only include existing data if not in incremental_only mode
                fetched_dfs.append(existing_df)

            for fetch_range in fetch_ranges:
                self.logger.info(
                    f"Panel {panel_id}: Fetching {fetch_range['reason']} range "
                    f"from {fetch_range['start']} to {fetch_range['end'] or 'now'}"
                )
                new_df = self.fetch_panel_data(
                    panel_id,
                    start_date_override=fetch_range["start"],
                    end_date_override=fetch_range["end"],
                )
                if len(new_df) > 0:
                    fetched_dfs.append(new_df)
                    self.logger.info(f"Panel {panel_id}: Fetched {len(new_df)} rows for {fetch_range['reason']} range")
                else:
                    self.logger.info(f"Panel {panel_id}: No new data in {fetch_range['reason']} range")

            # Merge all DataFrames
            if len(fetched_dfs) == 0:
                raise ValueError(f"No data available for panel {panel_id} in requested range")
            elif len(fetched_dfs) == 1:
                panel_data[panel_id] = fetched_dfs[0]
            else:
                combined = pd.concat(fetched_dfs)
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                panel_data[panel_id] = combined
                if incremental_only:
                    self.logger.info(f"Panel {panel_id}: Fetched {len(combined)} new rows (incremental mode)")
                else:
                    self.logger.info(f"Panel {panel_id}: Combined data has {len(combined)} total rows")

        self.logger.info("Data fetch completed successfully")
        return panel_data
