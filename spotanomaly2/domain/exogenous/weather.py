"""Weather as an exogenous source: fetcher + joiner pair (Open-Meteo API)."""

import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException

from spotanomaly2.infrastructure import logging as infra_logging


class ExogenousWeatherFetcher:
    """Fetcher for the weather exogenous source: pulls Open-Meteo data into a Parquet cache.

    Implements the ``ExogenousFetcher`` protocol. Open-Meteo provides free weather
    data without an API key; supports historical (Archive) and future (Forecast)
    endpoints with cache-aware range extension.
    """

    # API URLs
    ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Weather parameters available in both Archive and Forecast APIs
    # Only these parameters should be used for ML training to ensure
    # consistency between training (historical data) and prediction (forecasts)
    HOURLY_PARAMS = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "snow_depth",
        "weather_code",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
    ]

    @classmethod
    def is_enabled(cls, source_config: dict[str, Any]) -> bool:
        if not source_config.get("enabled", True):
            return False
        return source_config.get("latitude") is not None and source_config.get("longitude") is not None

    def __init__(
        self,
        source_config: dict[str, Any],
        parent_config: dict[str, Any],
        logger=None,
    ):
        """Args:
        source_config: The YAML ``config:`` sub-block for this source
            (``latitude``, ``longitude``, ``use_forecast``, ``lookback_days``,
            ``cache_path``, ``feature_columns``, ``fallback_on_failure``).
        parent_config: The full application config (kept for symmetry with
            other fetchers; not used at fetch time).
        logger: Optional logger instance.
        """
        self.source_config = source_config
        self.parent_config = parent_config
        self.logger = logger or infra_logging.get_logger("ExogenousWeatherFetcher")

        self.latitude = source_config.get("latitude")
        self.longitude = source_config.get("longitude")
        self.use_forecast = source_config.get("use_forecast", True)

        if self.latitude is None or self.longitude is None:
            raise ValueError("weather source_config must set latitude and longitude")

    def _fetch_from_api(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        api_type: str = "archive",
        timezone: str = "UTC",
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch weather data from Open-Meteo API (Archive or Forecast) with retry logic.

        Args:
            start: Start date/time (format: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM').
            end: End date/time (format: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM').
            api_type: Type of API to use - 'archive' or 'forecast' (default: 'archive').
            timezone: Timezone for the data (default: 'UTC').
            max_retries: Maximum number of retry attempts (default: 3).
            initial_backoff: Initial backoff time in seconds (default: 1.0).

        Returns:
            DataFrame with datetime index and weather parameter columns.

        Raises:
            requests.exceptions.RequestException: If API request fails after all retries.
            ValueError: If API returns an error or invalid api_type.
        """
        if api_type not in ("archive", "forecast"):
            raise ValueError(f"api_type must be 'archive' or 'forecast', got: {api_type}")

        api_name = "Archive" if api_type == "archive" else "Forecast"
        base_url = self.ARCHIVE_BASE_URL if api_type == "archive" else self.FORECAST_BASE_URL

        self.logger.info(f"Fetching weather data from {api_name} API: {start} to {end} (timezone: {timezone})")

        # Convert to pandas Timestamp for consistent handling
        if not isinstance(start, pd.Timestamp):
            start = pd.Timestamp(start)
        if not isinstance(end, pd.Timestamp):
            end = pd.Timestamp(end)

        # Ensure timezone-aware for forecast
        if api_type == "forecast":
            if start.tz is None:
                start = start.tz_localize(timezone)
            if end.tz is None:
                end = end.tz_localize(timezone)

        # Build API request parameters
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(self.HOURLY_PARAMS),
            "timezone": timezone,
        }

        # Add API-specific parameters
        if api_type == "archive":
            start_date = start.strftime("%Y-%m-%d")
            end_date = end.strftime("%Y-%m-%d")
            params["start_date"] = start_date
            params["end_date"] = end_date
        else:  # forecast
            now = pd.Timestamp.now(tz=timezone)
            days_ahead = (end - now).days + 2  # Add buffer
            days_ahead = min(max(1, days_ahead), 16)  # Clamp to 1-16
            params["forecast_days"] = days_ahead

        self.logger.debug(f"{api_name} API request params: {params}")

        # Retry loop with exponential backoff
        for attempt in range(max_retries + 1):
            try:
                # Make API request
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Success - break out of retry loop
                break

            except (HTTPError, RequestException) as e:
                if attempt == max_retries:
                    # Final attempt failed, re-raise the exception
                    self.logger.error(f"Weather {api_name} API request failed after {max_retries + 1} attempts: {e}")
                    raise

                # Calculate exponential backoff
                wait_time = initial_backoff * (2**attempt)
                self.logger.warning(
                    f"Weather {api_name} API call failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)

        # Check for API errors
        if "error" in data and data["error"]:
            raise ValueError(f"Open-Meteo {api_name} API error: {data.get('reason', 'Unknown error')}")

        # Parse response into DataFrame
        hourly_data = data.get("hourly", {})
        if not hourly_data:
            raise ValueError(f"No hourly data returned from {api_name} API")

        # Extract time and convert to datetime
        times = pd.to_datetime(hourly_data["time"])

        # Create DataFrame with weather parameters
        df_dict = {"datetime": times}
        for param in self.HOURLY_PARAMS:
            if param in hourly_data:
                df_dict[param] = hourly_data[param]

        df = pd.DataFrame(df_dict)
        df.set_index("datetime", inplace=True)

        # Convert timezone if needed
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone)
        if timezone != "UTC":
            df.index = df.index.tz_convert("UTC")

        # Filter to requested range for forecast API
        if api_type == "forecast":
            start_utc = start.tz_convert("UTC") if start.tz else start
            end_utc = end.tz_convert("UTC") if end.tz else end
            df = df.loc[start_utc:end_utc]

        self.logger.info(f"Successfully fetched {len(df)} hours of weather data from {api_name} API")
        self.logger.debug(f"Data range: {df.index.min()} to {df.index.max()}")

        return df

    def fetch_weather_data(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        timezone: str = "UTC",
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch weather data from Open-Meteo Archive API with retry logic.

        Args:
            start: Start date/time (format: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM').
            end: End date/time (format: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM').
            timezone: Timezone for the data (default: 'UTC').
            max_retries: Maximum number of retry attempts (default: 3).
            initial_backoff: Initial backoff time in seconds (default: 1.0).

        Returns:
            DataFrame with datetime index and weather parameter columns.

        Raises:
            requests.exceptions.RequestException: If API request fails after all retries.
            ValueError: If API returns an error.
        """
        return self._fetch_from_api(start, end, "archive", timezone, max_retries, initial_backoff)

    def fetch_forecast_data(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        timezone: str = "UTC",
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch weather forecast from Open-Meteo Forecast API with retry logic.

        Args:
            start: Start date/time (must be >= now, format: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM').
            end: End date/time (max 16 days from now).
            timezone: Timezone for the data (default: 'UTC').
            max_retries: Maximum number of retry attempts (default: 3).
            initial_backoff: Initial backoff time in seconds (default: 1.0).

        Returns:
            DataFrame with datetime index and weather parameter columns.

        Raises:
            requests.exceptions.RequestException: If API request fails after all retries.
            ValueError: If API returns an error or date range is invalid.
        """
        return self._fetch_from_api(start, end, "forecast", timezone, max_retries, initial_backoff)

    @staticmethod
    def load_from_cache(cache_path: Path) -> pd.DataFrame:
        """Load weather data from Parquet cache file.

        Args:
            cache_path: Path to the Parquet cache file.

        Returns:
            DataFrame with datetime index and weather columns.

        Raises:
            FileNotFoundError: If cache file does not exist.
        """
        logger = infra_logging.get_logger("WeatherFetcher")
        logger.debug(f"Attempting to load weather cache from: {cache_path}")

        if not cache_path.exists():
            logger.debug(f"Cache file not found: {cache_path}")
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        df = pd.read_parquet(cache_path)

        # Ensure index is datetime and timezone-aware
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df.set_index("datetime", inplace=True)
            else:
                raise ValueError("Cache file must have a datetime index or 'datetime' column")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        logger.info(f"Loaded {len(df)} records from cache: {cache_path}")
        logger.debug(f"Cache data range: {df.index.min()} to {df.index.max()}")

        return df

    @staticmethod
    def save_to_cache(cache_path: Path, df: pd.DataFrame) -> None:
        """Save weather data to Parquet cache file with compression.

        Args:
            cache_path: Path where the Parquet file will be saved.
            df: DataFrame to save.
        """
        logger = infra_logging.get_logger("WeatherFetcher")
        logger.debug(f"Saving weather cache to: {cache_path}")

        # Create parent directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with compression
        df.to_parquet(cache_path, compression="snappy", index=True)

        logger.info(f"Saved {len(df)} records to cache: {cache_path}")
        logger.debug(f"Cached data range: {df.index.min()} to {df.index.max()}")

    def _create_fallback_data(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        last_successful_data: pd.DataFrame,
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        """Repeat last 24 hours of weather data to fill missing range.

        When API calls fail, this method creates fallback data by repeating
        the last 24 hours of successfully fetched data in a cyclic pattern.

        Args:
            start: Start timestamp for the required data range.
            end: End timestamp for the required data range.
            last_successful_data: DataFrame containing at least 24 hours of valid data.
            timezone: Timezone for the data (default: 'UTC').

        Returns:
            DataFrame with weather data for the requested range, created by
            repeating the last 24 hours of available data.

        Raises:
            ValueError: If last_successful_data has fewer than 24 hours of data.
        """
        self.logger.warning(f"Creating fallback weather data for range {start} to {end}")
        self.logger.debug(f"Using last 24 hours from {len(last_successful_data)} available records")

        # Ensure we have at least 24 hours of data
        if len(last_successful_data) < 24:
            self.logger.error(f"Insufficient data for fallback: need 24 hours, got {len(last_successful_data)} hours")
            raise ValueError(f"Need at least 24 hours of data for fallback, got {len(last_successful_data)} hours")

        # Get the last 24 hours
        last_24h = last_successful_data.tail(24).copy()

        # Calculate how many hours we need to generate
        hours_needed = int((end - start).total_seconds() / 3600) + 1

        # Calculate how many times we need to repeat the 24-hour pattern
        num_repeats = (hours_needed // 24) + 1

        # Repeat the 24-hour pattern
        repeated_data = pd.concat([last_24h] * num_repeats, ignore_index=True)

        # Create new datetime index starting from 'start'
        new_index = pd.date_range(start=start, periods=hours_needed, freq="h", tz=timezone)

        # Assign the new index (trim to exact length needed)
        repeated_data = repeated_data.iloc[:hours_needed].copy()
        repeated_data.index = new_index

        # Convert to UTC if needed
        if timezone != "UTC" and repeated_data.index.tz is not None:
            repeated_data.index = repeated_data.index.tz_convert("UTC")

        self.logger.info(
            f"Created {len(repeated_data)} hours of fallback weather data by repeating the last 24-hour pattern"
        )

        return repeated_data

    def _split_historical_future(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[tuple[pd.Timestamp, pd.Timestamp] | None, tuple[pd.Timestamp, pd.Timestamp] | None]:
        """Split date range into historical and future parts.

        The Archive API typically has data up to ~5 days ago, while the Forecast API
        provides predictions for up to 16 days ahead from now.

        Args:
            start: Start timestamp (timezone-aware).
            end: End timestamp (timezone-aware).

        Returns:
            Tuple of (historical_range, future_range) where each is either
            (start_ts, end_ts) or None if that range doesn't exist.
        """
        now = pd.Timestamp.now(tz=start.tz)
        # Archive API typically has reliable coverage only up to ~5 days ago.
        # Newer timestamps (including recent past) are handled via forecast API.
        archive_cutoff = now - pd.Timedelta(days=5)

        self.logger.debug(f"Splitting date range: now={now}, archive_cutoff={archive_cutoff}")

        historical_range = None
        future_range = None

        if end <= archive_cutoff:
            # Entirely historical
            historical_range = (start, end)
            self.logger.debug(f"Date range is entirely historical: {start} to {end}")
        elif start >= archive_cutoff:
            # Entirely in the recent/forecast-supported range (recent past and/or future)
            future_range = (start, end)
            self.logger.debug(f"Date range is entirely forecast-supported: {start} to {end}")
        else:
            # Spans historical archive-covered range and forecast-supported range
            if start < archive_cutoff:
                historical_range = (start, min(archive_cutoff, end))
                self.logger.debug(f"Historical portion: {historical_range[0]} to {historical_range[1]}")
            if end > archive_cutoff:
                future_range = (max(archive_cutoff, start), end)
                self.logger.debug(f"Future portion: {future_range[0]} to {future_range[1]}")

        return historical_range, future_range

    def fetch_weather_data_hybrid(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        timezone: str = "UTC",
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch weather data using appropriate API(s) based on date range.

        Automatically determines which API(s) to use:
        - Archive API for historical dates (up to ~5 days ago)
        - Forecast API for future dates (up to 16 days ahead)
        - Both APIs if the range spans historical and future

        Args:
            start: Start date/time.
            end: End date/time.
            timezone: Timezone for the data (default: 'UTC').
            max_retries: Maximum number of retry attempts (default: 3).
            initial_backoff: Initial backoff time in seconds (default: 1.0).

        Returns:
            DataFrame with datetime index and weather columns for the requested range.

        Raises:
            ValueError: If both APIs fail or if forecast is disabled and range includes future dates.
        """
        self.logger.info(f"Fetching weather data (hybrid mode): {start} to {end}")

        # Convert to pandas Timestamp for consistent handling
        if not isinstance(start, pd.Timestamp):
            start = pd.Timestamp(start)
        if not isinstance(end, pd.Timestamp):
            end = pd.Timestamp(end)

        # Ensure timezone-aware
        if start.tz is None:
            start = start.tz_localize(timezone)
        if end.tz is None:
            end = end.tz_localize(timezone)

        # Convert to UTC for internal processing
        start_utc = start.tz_convert("UTC")
        end_utc = end.tz_convert("UTC")

        # Split into historical and future ranges
        hist_range, future_range = self._split_historical_future(start, end)

        dfs = []

        # Fetch historical data if needed
        if hist_range:
            try:
                self.logger.info(f"Fetching historical weather data from {hist_range[0]} to {hist_range[1]}")
                df_hist = self.fetch_weather_data(hist_range[0], hist_range[1], timezone, max_retries, initial_backoff)
                dfs.append(df_hist)
            except Exception as e:
                self.logger.warning(f"Failed to fetch historical weather data: {e}")
                # Don't raise yet, maybe forecast will work

        # Fetch forecast data if needed and enabled
        if future_range:
            if self.use_forecast:
                try:
                    self.logger.info(f"Fetching forecast weather data from {future_range[0]} to {future_range[1]}")
                    df_forecast = self.fetch_forecast_data(
                        future_range[0], future_range[1], timezone, max_retries, initial_backoff
                    )
                    dfs.append(df_forecast)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch forecast weather data: {e}")
                    # Forecast failed, will rely on fallback mechanism later
            else:
                self.logger.info("Forecast API disabled, will use fallback for future dates if available")

        # Merge results
        if not dfs:
            self.logger.error("Failed to fetch data from all applicable APIs")
            raise ValueError("Failed to fetch data from all applicable APIs")

        self.logger.debug(f"Merging {len(dfs)} dataframes from different API sources")
        merged = pd.concat(dfs).sort_index()

        # Remove duplicates, keeping first occurrence
        merged = merged[~merged.index.duplicated(keep="first")]

        # Align to requested range
        result = merged.loc[start_utc:end_utc]

        self.logger.info(f"Successfully fetched {len(result)} hours of hybrid weather data")
        self.logger.debug(f"Final data range: {result.index.min()} to {result.index.max()}")

        return result

    def get_weather_data(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        cache_path: Optional[Path] = None,
        timezone: str = "UTC",
        max_retries: int = 3,
        fallback_on_failure: bool = True,
    ) -> pd.DataFrame:
        """Get weather data with cache-first strategy, retry logic, and fallback.

        First attempts to load from cache. If cache doesn't exist or doesn't
        cover the requested date range, fetches missing data from API with
        automatic retries. If all retries fail and fallback is enabled, repeats
        the last 24 hours of available data.

        Args:
            start: Start date/time.
            end: End date/time.
            cache_path: Path to cache file (optional). If None, always fetches from API.
            timezone: Timezone for the data (default: 'UTC').
            max_retries: Maximum number of retry attempts for API calls (default: 3).
            fallback_on_failure: If True, use 24-hour repetition fallback when API fails (default: True).

        Returns:
            DataFrame with datetime index and weather columns for the requested range.

        Raises:
            Exception: If API fails and fallback is disabled or not possible.
        """
        self.logger.info(f"Getting weather data: {start} to {end} (timezone: {timezone})")
        if cache_path:
            self.logger.debug(f"Using cache: {cache_path}")
        else:
            self.logger.debug("No cache specified, will fetch from API")

        # Convert to pandas Timestamp for easier comparison
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # Ensure timezone-aware
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize(timezone)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize(timezone)

        # Convert to UTC for internal processing
        start_utc = start_ts.tz_convert("UTC")
        end_utc = end_ts.tz_convert("UTC")

        # Try to load from cache
        cached_df = None
        if cache_path is not None:
            try:
                cached_df = self.load_from_cache(cache_path)
            except FileNotFoundError:
                self.logger.info("Cache file not found, will fetch from API")

        # Check if cache covers the requested range
        if cached_df is not None:
            cache_start = cached_df.index.min()
            cache_end = cached_df.index.max()

            self.logger.debug(f"Cache covers: {cache_start} to {cache_end}")
            self.logger.debug(f"Requested range: {start_utc} to {end_utc}")

            # If cache fully covers the range, return subset
            if cache_start <= start_utc and cache_end >= end_utc:
                self.logger.info("Cache fully covers requested range, using cached data")
                result = cached_df.loc[start_utc:end_utc]
                self.logger.debug(f"Returning {len(result)} hours from cache")
                return result

            # Cache exists but doesn't cover full range - fetch missing data
            self.logger.info("Cache does not fully cover requested range, fetching missing data")

            # Determine which ranges to fetch
            fetch_ranges = []

            if start_utc < cache_start:
                fetch_ranges.append((start_utc, cache_start - pd.Timedelta(hours=1)))
                self.logger.debug(f"Need to fetch before cache: {start_utc} to {cache_start - pd.Timedelta(hours=1)}")

            if end_utc > cache_end:
                fetch_ranges.append((cache_end + pd.Timedelta(hours=1), end_utc))
                self.logger.debug(f"Need to fetch after cache: {cache_end + pd.Timedelta(hours=1)} to {end_utc}")

            # Fetch missing data and merge with cache
            new_dfs = [cached_df]
            for fetch_start, fetch_end in fetch_ranges:
                self.logger.info(f"Fetching missing range: {fetch_start} to {fetch_end}")
                try:
                    new_df = self.fetch_weather_data_hybrid(fetch_start, fetch_end, timezone, max_retries=max_retries)
                    new_dfs.append(new_df)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch weather data for range {fetch_start} to {fetch_end}: {e}")

                    # Use fallback if enabled
                    if fallback_on_failure:
                        try:
                            fallback_df = self._create_fallback_data(fetch_start, fetch_end, cached_df, timezone)
                            new_dfs.append(fallback_df)
                        except Exception as fallback_error:
                            self.logger.error(
                                f"Fallback also failed for range {fetch_start} to {fetch_end}: {fallback_error}"
                            )
                            # Continue without this range
                    else:
                        # Fallback disabled, re-raise the exception
                        raise

            # Merge all dataframes and sort by index
            self.logger.debug(f"Merging {len(new_dfs)} dataframes (cache + new data)")
            merged_df = pd.concat(new_dfs).sort_index()

            # Remove duplicates, keeping first occurrence
            merged_df = merged_df[~merged_df.index.duplicated(keep="first")]

            # Save updated cache
            if cache_path is not None:
                self.logger.info("Updating cache with newly fetched data")
                self.save_to_cache(cache_path, merged_df)

            result = merged_df.loc[start_utc:end_utc]
            self.logger.info(f"Returning {len(result)} hours of weather data (cache + API)")
            return result

        # No cache, try to fetch from API (hybrid method)
        self.logger.info("No cache available, fetching all data from API")
        try:
            df = self.fetch_weather_data_hybrid(start, end, timezone, max_retries=max_retries)
            if cache_path is not None:
                self.logger.info("Creating new cache file")
                self.save_to_cache(cache_path, df)
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch weather data from API: {e}")

            # No cache and API failed - cannot use fallback
            if not fallback_on_failure:
                raise

            # Fallback requires existing data, which we don't have
            self.logger.error("Cannot use fallback: no cached data available and API request failed")
            raise ValueError(
                "Weather API failed and no cached data available for fallback. "
                "Please check your internet connection or try again later."
            ) from e

    def run(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        cache_path: Optional[Path] = None,
        timezone: str = "UTC",
        max_retries: int = 3,
        fallback_on_failure: bool = True,
    ) -> pd.DataFrame:
        """Run the complete weather fetch pipeline.

        Args:
            start: Start date/time.
            end: End date/time.
            cache_path: Path to cache file (optional).
            timezone: Timezone for the data (default: 'UTC').
            max_retries: Maximum number of retry attempts (default: 3).
            fallback_on_failure: If True, use fallback when API fails (default: True).

        Returns:
            DataFrame with datetime index and weather columns.
        """
        self.logger.info("Starting weather data fetch...")
        df = self.get_weather_data(
            start=start,
            end=end,
            cache_path=cache_path,
            timezone=timezone,
            max_retries=max_retries,
            fallback_on_failure=fallback_on_failure,
        )
        self.logger.info("Weather data fetch completed successfully")
        return df

    # ------------------------------------------------------------------
    # ExogenousFetcher protocol entry point
    # ------------------------------------------------------------------

    def fetch_and_cache(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Fetch weather for ``[start - lookback_days, end]`` and persist to ``cache_path``.

        The lookback expansion lives here (not in the orchestrator) because the
        baseline column the joiner computes needs ``lookback_days`` of pre-window
        history.
        """
        lookback_days = self.source_config.get("lookback_days", 14)
        cache_path_str = self.source_config.get("cache_path")
        cache_path = Path(cache_path_str) if cache_path_str else None
        fallback_on_failure = self.source_config.get("fallback_on_failure", True)

        fetch_start = start - pd.Timedelta(days=lookback_days)
        self.logger.info(f"Fetching weather: {fetch_start} to {end} (lookback_days={lookback_days})")
        self.get_weather_data(
            start=fetch_start,
            end=end,
            cache_path=cache_path,
            timezone="UTC",
            fallback_on_failure=fallback_on_failure,
        )


class ExogenousWeatherJoiner:
    """Joiner for the weather exogenous source: loads cached weather + computes baseline.

    Implements the ``ExogenousJoiner`` protocol. Reads the parquet written by
    ``ExogenousWeatherFetcher.fetch_and_cache``, computes a rolling
    ``lookback_days`` baseline over ``temperature_2m`` (since the baseline needs
    pre-panel history a per-panel index can't carry), and reindexes
    ``weather_temperature_baseline`` plus configured ``weather_<feature>``
    columns onto each panel.
    """

    def __init__(
        self,
        source_config: dict[str, Any],
        parent_config: dict[str, Any],
        logger=None,
    ):
        self.source_config = source_config
        self.parent_config = parent_config
        self.logger = logger or infra_logging.get_logger("ExogenousWeatherJoiner")

    def _load_cached_weather(self) -> Optional[pd.DataFrame]:
        cache_path_str = self.source_config.get("cache_path")
        if not cache_path_str:
            self.logger.warning("Weather cache_path not configured; skipping join")
            return None
        cache_path = Path(cache_path_str)
        if not cache_path.exists():
            self.logger.warning(f"Weather cache {cache_path} not found; skipping join")
            return None
        df = pd.read_parquet(cache_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning(f"Weather cache {cache_path} is not datetime-indexed; skipping join")
            return None
        return df

    def join_into_panels(
        self,
        panel_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Read cached weather and join baseline + feature columns onto panels."""
        if not panel_data:
            return panel_data

        non_empty = {pid: df for pid, df in panel_data.items() if len(df) > 0}
        if not non_empty:
            return panel_data

        weather_df = self._load_cached_weather()
        if weather_df is None:
            return panel_data

        lookback_days = self.source_config.get("lookback_days", 14)
        feature_columns = self.source_config.get("feature_columns", [])
        panel_freq = self.parent_config.get("process", {}).get("resample", {}).get("freq", "5min")

        baseline_full: Optional[pd.Series] = None
        if "temperature_2m" in weather_df.columns:
            temperature_resampled = weather_df["temperature_2m"].resample(panel_freq).ffill()
            baseline_full = temperature_resampled.rolling(window=f"{lookback_days}D", min_periods=1).mean()
            if baseline_full.isna().any():
                baseline_full = baseline_full.fillna(temperature_resampled.mean())
        else:
            self.logger.warning("temperature_2m not in cached weather; skipping baseline column")

        resampled_features: dict[str, pd.Series] = {}
        for col in feature_columns:
            if col in weather_df.columns:
                resampled_features[f"weather_{col}"] = weather_df[col].resample(panel_freq).ffill()
            else:
                self.logger.warning(f"Configured weather feature_column '{col}' not in cached weather")

        merged: dict[str, pd.DataFrame] = {}
        for panel_id, df in panel_data.items():
            if len(df) == 0:
                merged[panel_id] = df
                continue
            out = df.copy()
            if baseline_full is not None:
                out["weather_temperature_baseline"] = baseline_full.reindex(df.index, method="ffill")
            for col_name, series in resampled_features.items():
                out[col_name] = series.reindex(df.index, method="ffill")
            merged[panel_id] = out

        self.logger.info(f"Joined weather onto {len(merged)} panel(s) using {panel_freq} grid")
        return merged
