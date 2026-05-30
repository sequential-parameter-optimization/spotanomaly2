"""Exogenous source A: fetcher + joiner pair (Parquet-cached)."""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException

from spotanomaly2.infrastructure import logging as infra_logging

_WIRING_TEMPLATE = {
    "dashboard_positionings": [],
    "input_wirings": [
        {
            "adapter_id": "virtual-structure-adapter",
            "filters": {
                "timestampFrom": "",
                "timestampTo": "",
            },
            "ref_id": "",
            "ref_id_type": "SOURCE",
            "type": "multitsframe",
            "use_default_value": False,
            "workflow_input_name": "input",
        }
    ],
    "output_wirings": [],
}


class ExogenousAFetcher:
    """Fetcher for exogenous source A: pulls timeseries data and writes to its Parquet cache.

    Implements the ``ExogenousFetcher`` protocol (see ``domain/exogenous/base.py``).
    Authenticates via Keycloak (OpenID Connect password grant), executes a transformation
    in the upstream system's virtual-structure-adapter, and persists per-source parquets
    under ``source_config["cache_dir"]``. Cache-aware: re-running over an overlapping
    range fetches only the missing slices.
    """

    @classmethod
    def is_enabled(cls, source_config: dict[str, Any]) -> bool:
        return bool(source_config.get("enabled", True))

    def __init__(
        self,
        source_config: dict[str, Any],
        parent_config: dict[str, Any],
        logger=None,
    ):
        """Args:
        source_config: The YAML ``config:`` sub-block for this source
            (``cache_dir``, ``panel_sources``, etc.).
        parent_config: The full application config — used to locate the
            credentials env file via ``paths.credentials_file``.
        logger: Optional logger instance.
        """
        self.source_config = source_config
        self.parent_config = parent_config
        self.logger = logger or infra_logging.get_logger("ExogenousAFetcher")

        self.cache_dir = Path(source_config.get("cache_dir", "data/exogenous"))

        from dotenv import load_dotenv

        creds_path = parent_config["paths"]["credentials_file"]
        load_dotenv(creds_path)
        load_dotenv()

        endpoint_url = os.getenv("EXOGENOUS_ENDPOINT_URL")
        token_url = os.getenv("EXOGENOUS_TOKEN_URL")
        trafo_id = os.getenv("EXOGENOUS_TRAFO_ID")
        client_id = os.getenv("EXOGENOUS_CLIENT_ID") or "exogenous-adapter"
        sources_raw = os.getenv("EXOGENOUS_SOURCES")

        missing = []
        if not endpoint_url:
            missing.append("EXOGENOUS_ENDPOINT_URL")
        if not token_url:
            missing.append("EXOGENOUS_TOKEN_URL")
        if not trafo_id:
            missing.append("EXOGENOUS_TRAFO_ID")
        if not sources_raw:
            missing.append("EXOGENOUS_SOURCES")
        if missing:
            raise ValueError(f"{', '.join(missing)} must be set in the credentials file ({creds_path}) or .env")

        try:
            sources = json.loads(sources_raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f'EXOGENOUS_SOURCES must be a JSON object (e.g. {{"source_a":"<uuid>","source_b":"<uuid>"}}): {e}'
            ) from e
        if not isinstance(sources, dict) or not all(isinstance(v, str) for v in sources.values()):
            raise ValueError("EXOGENOUS_SOURCES must be a JSON object mapping source names to ref UUID strings")

        self.endpoint_url = endpoint_url
        self.token_url = token_url
        self.trafo_id = trafo_id
        self.client_id = client_id
        self.sources = sources

        self._username: str | None = None
        self._password: str | None = None
        self._bearer_token: str | None = None
        self._token_expires_at: float = 0

    # ------------------------------------------------------------------
    # Credential helpers
    # ------------------------------------------------------------------

    def _load_credentials(self) -> tuple[str, str]:
        """Load EXOGENOUS credentials from the env file once."""
        if self._username is not None and self._password is not None:
            return self._username, self._password

        from dotenv import load_dotenv

        creds_path = self.parent_config["paths"]["credentials_file"]
        load_dotenv(creds_path)
        load_dotenv()  # fallback: root .env

        username = os.getenv("EXOGENOUS_USERNAME")
        password = os.getenv("EXOGENOUS_PASSWORD")

        if not username or not password:
            raise ValueError(
                f"EXOGENOUS_USERNAME and EXOGENOUS_PASSWORD must be set in the credentials file ({creds_path}) or .env"
            )

        self._username = username
        self._password = password
        return username, password

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _acquire_token(self, max_retries: int = 3, initial_backoff: float = 1.0) -> str:
        """Acquire a bearer token from the Keycloak token endpoint.

        Returns:
            The full ``"Bearer <token>"`` string.
        """
        username, password = self._load_credentials()

        body = {
            "grant_type": "password",
            "client_id": self.client_id,
            "username": username,
            "password": password,
        }

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(self.token_url, data=body, timeout=30)
                response.raise_for_status()
                data = response.json()

                expires_in = data.get("expires_in", 300)
                self._token_expires_at = time.time() + expires_in - 60
                self._bearer_token = f"Bearer {data['access_token']}"

                self.logger.info("EXOGENOUS bearer token acquired")
                return self._bearer_token

            except (HTTPError, RequestException) as exc:
                if attempt == max_retries:
                    self.logger.error(f"Token request failed after {max_retries + 1} attempts: {exc}")
                    raise

                wait = initial_backoff * (2**attempt)
                self.logger.warning(
                    f"Token request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait}s: {exc}"
                )
                time.sleep(wait)

        raise RuntimeError("Unreachable")

    def _ensure_valid_token(self) -> str:
        """Return a valid bearer token, refreshing if expired."""
        if self._bearer_token is None or time.time() >= self._token_expires_at:
            return self._acquire_token()
        return self._bearer_token

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_wiring(ref_id: str, start: str, end: str) -> dict:
        """Build the adapter wiring payload for a single source.

        Args:
            ref_id: UUID of the virtual-structure-adapter source.
            start: ISO-8601 start timestamp (nanosecond precision accepted).
            end: ISO-8601 end timestamp.

        Returns:
            A wiring dict ready to be POSTed to the execute endpoint.
        """
        wiring = {
            "dashboard_positionings": list(_WIRING_TEMPLATE["dashboard_positionings"]),
            "input_wirings": [
                {
                    **_WIRING_TEMPLATE["input_wirings"][0],
                    "filters": {
                        "timestampFrom": start,
                        "timestampTo": end,
                    },
                    "ref_id": ref_id,
                }
            ],
            "output_wirings": list(_WIRING_TEMPLATE["output_wirings"]),
        }
        return wiring

    def _fetch_timeseries(
        self,
        ref_id: str,
        start: str,
        end: str,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> pd.DataFrame:
        """Execute a Exogenous transformation and return the output as a DataFrame.

        Args:
            ref_id: Virtual-structure-adapter source UUID.
            start: ISO-8601 start timestamp.
            end: ISO-8601 end timestamp.
            max_retries: Maximum retry attempts on transient failures.
            initial_backoff: Initial backoff in seconds (doubled each retry).

        Returns:
            DataFrame with a DatetimeIndex (UTC).

        Raises:
            RequestException: If all retries are exhausted.
            ValueError: If the response cannot be parsed.
        """
        token = self._ensure_valid_token()
        wiring = self._build_wiring(ref_id, start, end)
        payload = {"id": self.trafo_id, "wiring": wiring}

        self.logger.info(f"Fetching EXOGENOUS timeseries for source {ref_id}: {start} -> {end}")

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint_url,
                    json=payload,
                    headers={"Authorization": token},
                    timeout=60,
                )
                response.raise_for_status()
                break
            except (HTTPError, RequestException) as exc:
                if attempt == max_retries:
                    self.logger.error(f"EXOGENOUS API request failed after {max_retries + 1} attempts: {exc}")
                    raise
                wait = initial_backoff * (2**attempt)
                self.logger.warning(
                    f"EXOGENOUS API call failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait}s: {exc}"
                )
                time.sleep(wait)

        result_json = response.json()
        output_data = result_json.get("output_results_by_output_name", {}).get("output", {}).get("__data__", {})
        if not output_data:
            raise ValueError(f"EXOGENOUS API returned no data for source {ref_id} ({start} -> {end})")

        df = pd.DataFrame.from_dict(output_data)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
        elif df.index.dtype == object:
            df.index = pd.to_datetime(df.index, utc=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Could not parse a DatetimeIndex from EXOGENOUS response for source {ref_id}")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # The exogenous multitsframe returns values as strings; coerce to numeric.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.sort_index(inplace=True)
        self.logger.info(f"Fetched {len(df)} rows for source {ref_id} ({df.index.min()} to {df.index.max()})")
        return df

    # ------------------------------------------------------------------
    # Public fetch (both sources)
    # ------------------------------------------------------------------

    def fetch_data(
        self,
        start: str,
        end: str,
        max_retries: int = 3,
    ) -> dict[str, pd.DataFrame]:
        """Fetch timeseries for all configured sources.

        Args:
            start: ISO-8601 start timestamp.
            end: ISO-8601 end timestamp.
            max_retries: Maximum retry attempts per source.

        Returns:
            Dict mapping source name to DataFrame (e.g. ``{"source_a": df, "source_b": df}``).
        """
        results: dict[str, pd.DataFrame] = {}
        for name, ref_id in self.sources.items():
            results[name] = self._fetch_timeseries(ref_id, start, end, max_retries=max_retries)
        return results

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_cache(cache_path: Path) -> pd.DataFrame:
        """Load timeseries data from a Parquet cache file.

        Args:
            cache_path: Path to the Parquet file.

        Returns:
            DataFrame with a UTC DatetimeIndex.

        Raises:
            FileNotFoundError: If the cache file does not exist.
        """
        logger = infra_logging.get_logger("ExogenousFetcher")
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        df = pd.read_parquet(cache_path)

        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("timestamp", "datetime"):
                if col in df.columns:
                    df.set_index(col, inplace=True)
                    break
            else:
                raise ValueError("Cache file must have a DatetimeIndex or a 'timestamp'/'datetime' column")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Loaded {len(df)} records from cache: {cache_path}")
        return df

    @staticmethod
    def save_to_cache(cache_path: Path, df: pd.DataFrame) -> None:
        """Save timeseries data to a Parquet cache file with snappy compression.

        Args:
            cache_path: Destination path.
            df: DataFrame to persist.
        """
        logger = infra_logging.get_logger("ExogenousFetcher")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, compression="snappy", index=True)
        logger.info(f"Saved {len(df)} records to cache: {cache_path}")

    # ------------------------------------------------------------------
    # Cache-first fetch with incremental extension
    # ------------------------------------------------------------------

    def _get_source_data(
        self,
        name: str,
        ref_id: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        cache_path: Path,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Get data for a single source with cache-first strategy.

        Loads cached data, determines which time ranges are missing, fetches
        only those from the API, merges everything, and updates the cache.

        Args:
            name: Human-readable source name (for logging).
            ref_id: Virtual-structure-adapter source UUID.
            start: Requested start.
            end: Requested end.
            cache_path: Path to the per-source Parquet cache file.
            max_retries: API retry attempts.

        Returns:
            DataFrame covering at least the requested range.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize("UTC")

        cached_df: Optional[pd.DataFrame] = None
        try:
            cached_df = self.load_from_cache(cache_path)
        except FileNotFoundError:
            self.logger.info(f"[{name}] No cache found at {cache_path}")

        if cached_df is not None:
            cache_start = cached_df.index.min()
            cache_end = cached_df.index.max()
            self.logger.debug(f"[{name}] Cache covers {cache_start} to {cache_end}, requested {start_ts} to {end_ts}")

            if cache_start <= start_ts and cache_end >= end_ts:
                self.logger.info(f"[{name}] Cache fully covers requested range")
                return cached_df.loc[start_ts:end_ts]

            fetch_ranges: list[tuple[str, str]] = []
            if start_ts < cache_start:
                fetch_ranges.append(
                    (
                        start_ts.isoformat(),
                        (cache_start - pd.Timedelta(seconds=1)).isoformat(),
                    )
                )
            if end_ts > cache_end:
                fetch_ranges.append(
                    (
                        (cache_end + pd.Timedelta(seconds=1)).isoformat(),
                        end_ts.isoformat(),
                    )
                )

            new_dfs = [cached_df]
            for fetch_start, fetch_end in fetch_ranges:
                self.logger.info(f"[{name}] Fetching missing range: {fetch_start} -> {fetch_end}")
                new_df = self._fetch_timeseries(ref_id, fetch_start, fetch_end, max_retries=max_retries)
                new_dfs.append(new_df)

            merged = pd.concat(new_dfs).sort_index()
            merged = merged[~merged.index.duplicated(keep="first")]

            self.save_to_cache(cache_path, merged)
            return merged.loc[start_ts:end_ts]

        self.logger.info(f"[{name}] Fetching full range from API")
        df = self._fetch_timeseries(ref_id, str(start), str(end), max_retries=max_retries)
        self.save_to_cache(cache_path, df)
        return df

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def get_data(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        max_retries: int = 3,
    ) -> dict[str, pd.DataFrame]:
        """Get timeseries for all configured sources using cache-first strategy.

        Args:
            start: Start timestamp (ISO-8601 or pd.Timestamp).
            end: End timestamp.
            max_retries: API retry attempts per source.

        Returns:
            Dict mapping source name to DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for name, ref_id in self.sources.items():
            cache_path = self.cache_dir / f"{name}.parquet"
            results[name] = self._get_source_data(name, ref_id, start, end, cache_path, max_retries=max_retries)
        return results

    def run(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        max_retries: int = 3,
    ) -> dict[str, pd.DataFrame]:
        """Run the complete Exogenous data fetch pipeline.

        Args:
            start: Start timestamp.
            end: End timestamp.
            max_retries: API retry attempts per source.

        Returns:
            Dict mapping source name to DataFrame.
        """
        self.logger.info("Starting Exogenous data fetch...")
        data = self.get_data(start=start, end=end, max_retries=max_retries)
        total_rows = sum(len(df) for df in data.values())
        self.logger.info(f"Exogenous data fetch completed: {total_rows} total rows across {len(data)} sources")
        return data

    # ------------------------------------------------------------------
    # ExogenousFetcher protocol entry point
    # ------------------------------------------------------------------

    def fetch_and_cache(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Fetch ``[start, end]`` for every configured source and persist to ``cache_dir``.

        Side effect only: ``get_data`` writes a parquet per source under
        ``cache_dir``. The downloader doesn't need the returned DataFrames.
        """
        self.run(start=start, end=end)


class ExogenousAJoiner:
    """Joiner for exogenous source A: loads cached parquets and merges columns onto panels.

    Implements the ``ExogenousJoiner`` protocol. Never triggers network I/O —
    if a cache file is missing it logs a warning and skips that source.
    """

    def __init__(
        self,
        source_config: dict[str, Any],
        parent_config: dict[str, Any],
        logger=None,
    ):
        self.source_config = source_config
        self.parent_config = parent_config
        self.logger = logger or infra_logging.get_logger("ExogenousAJoiner")

        self.cache_dir = Path(source_config.get("cache_dir", "data/exogenous"))
        self.panel_sources: dict[str, list[str]] = source_config.get("panel_sources", {})
        # YAML source identity, injected by the registry. Columns are prefixed
        # ``exogenous_<exo_name>_<cache_key>`` so they map back to this source.
        self.exo_name: str = source_config.get("source_name", "a")

    def _load_source_cache(self, source_name: str) -> Optional[pd.DataFrame]:
        cache_path = self.cache_dir / f"{source_name}.parquet"
        if not cache_path.exists():
            self.logger.warning(f"Exogenous cache for '{source_name}' not found at {cache_path}, skipping")
            return None
        df = pd.read_parquet(cache_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning(f"Exogenous cache for '{source_name}' is not datetime-indexed; skipping")
            return None
        # Drop metric column if present (kept for legacy parity with the old merge path).
        exclude_cols = {"metric"}
        keep = [c for c in df.columns if c.lower() not in exclude_cols]
        if not keep:
            self.logger.warning(f"Exogenous source '{source_name}' cache has no measurement columns, skipping")
            return None
        df_vals = df[keep].copy()
        if len(keep) == 1:
            df_vals.columns = [f"exogenous_{self.exo_name}_{source_name}"]
        else:
            df_vals.columns = [f"exogenous_{self.exo_name}_{source_name}_{c}" for c in keep]
        return df_vals

    def join_into_panels(
        self,
        panel_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Load per-source caches and concat their columns onto each panel."""
        if not panel_data:
            return panel_data

        # Discover sources we need across all panels.
        if self.panel_sources:
            needed = {s for sources in self.panel_sources.values() for s in sources}
        else:
            # No per-panel mapping: load whatever caches exist alongside config.
            needed = {p.stem for p in self.cache_dir.glob("*.parquet")} if self.cache_dir.exists() else set()

        source_frames: dict[str, pd.DataFrame] = {}
        for source_name in needed:
            df_vals = self._load_source_cache(source_name)
            if df_vals is not None:
                source_frames[source_name] = df_vals

        if not source_frames:
            self.logger.warning("No usable exogenous caches loaded; returning panel_data unchanged")
            return panel_data

        result: dict[str, pd.DataFrame] = {}
        for panel_id, df_panel in panel_data.items():
            if self.panel_sources:
                source_names = self.panel_sources.get(str(panel_id), [])
                frames = [source_frames[n] for n in source_names if n in source_frames]
                if not frames:
                    self.logger.warning(
                        f"Panel {panel_id}: no matching exogenous sources loaded "
                        f"(panel_sources={source_names}), skipping"
                    )
                    result[panel_id] = df_panel
                    continue
            else:
                frames = list(source_frames.values())

            exogenous_combined = pd.concat(frames, axis=1).sort_index()

            left = df_panel.sort_index()
            right = exogenous_combined

            # Normalise datetime resolution (us vs ns) so concat aligns.
            target_res = "ns"
            if hasattr(left.index.dtype, "tz"):
                left.index = left.index.as_unit(target_res)
            if hasattr(right.index.dtype, "tz"):
                right.index = right.index.as_unit(target_res)

            merged = pd.concat([left, right], axis=1).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            merged.index.name = df_panel.index.name

            exogenous_cols = [c for c in merged.columns if c not in df_panel.columns]
            self.logger.info(f"Panel {panel_id}: joined {len(exogenous_cols)} exogenous columns, {len(merged)} rows")
            result[panel_id] = merged

        return result
