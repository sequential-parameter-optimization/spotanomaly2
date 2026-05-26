# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP-mocked tests for :mod:`spotanomaly2.domain.weather_fetcher`.

The production class talks to the Open-Meteo Archive + Forecast APIs, retries
with exponential backoff, caches to parquet, and has a silent 24-hour cyclic
fallback for API failures.  Every test here patches ``weather_fetcher.requests.get``
so no real network call is made, and patches ``weather_fetcher.time.sleep`` so
retry tests complete in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from spotanomaly2.domain import weather_fetcher as wf_module
from spotanomaly2.domain.weather_fetcher import WeatherFetcher


# ---------------------------------------------------------------------------
# Helpers


def _hourly_payload(start: str, periods: int) -> dict[str, Any]:
    """Build a fake Open-Meteo ``hourly`` JSON payload."""
    times = pd.date_range(start=start, periods=periods, freq="h")
    return {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": [10.0 + i * 0.1 for i in range(periods)],
            "relative_humidity_2m": [50.0] * periods,
            "precipitation": [0.0] * periods,
            "rain": [0.0] * periods,
            "snowfall": [0.0] * periods,
            "snow_depth": [0.0] * periods,
            "weather_code": [0] * periods,
            "pressure_msl": [1013.0] * periods,
            "surface_pressure": [1010.0] * periods,
            "cloud_cover": [25] * periods,
            "cloud_cover_low": [10] * periods,
            "cloud_cover_mid": [10] * periods,
            "cloud_cover_high": [5] * periods,
            "wind_speed_10m": [3.0] * periods,
            "wind_direction_10m": [180] * periods,
            "wind_gusts_10m": [5.0] * periods,
        }
    }


def _mk_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Build a MagicMock that quacks like a ``requests.Response``."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}

    if status_code >= 400:
        http_err = requests.exceptions.HTTPError(f"{status_code} Server Error")
        http_err.response = resp
        resp.raise_for_status.side_effect = http_err
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture(autouse=True)
def _no_sleep():
    """Make ``time.sleep`` a no-op for every test in this module."""
    with patch.object(wf_module.time, "sleep", return_value=None) as m:
        yield m


@pytest.fixture
def cfg() -> dict[str, Any]:
    """Minimal location config the WeatherFetcher needs."""
    return {"latitude": 50.0, "longitude": 7.0, "use_forecast": True}


@pytest.fixture
def fetcher(cfg) -> WeatherFetcher:
    return WeatherFetcher(cfg)


# ---------------------------------------------------------------------------
# 1. Happy path


class TestFetchWeatherDataHappyPath:
    def test_returns_dataframe_with_expected_shape_and_tz(self, fetcher):
        payload = _hourly_payload("2025-01-01T00:00", periods=48)
        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, payload)) as mock_get:
            df = fetcher.fetch_weather_data("2025-01-01", "2025-01-02")

        assert mock_get.call_count == 1
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 48
        # tz-aware UTC index
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"
        # Every requested HOURLY_PARAM column should be present
        for col in WeatherFetcher.HOURLY_PARAMS:
            assert col in df.columns, f"missing column: {col}"

    def test_archive_endpoint_used_for_past_dates(self, fetcher):
        payload = _hourly_payload("2025-01-01T00:00", periods=24)
        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, payload)) as mock_get:
            fetcher.fetch_weather_data("2025-01-01", "2025-01-01")

        called_url = mock_get.call_args.args[0]
        assert called_url == WeatherFetcher.ARCHIVE_BASE_URL
        # Archive uses start_date / end_date params (not forecast_days)
        params = mock_get.call_args.kwargs["params"]
        assert "start_date" in params and "end_date" in params
        assert "forecast_days" not in params


# ---------------------------------------------------------------------------
# 2. Retry on 5xx


class TestRetryBehaviour:
    def test_retries_then_succeeds_on_5xx(self, fetcher):
        payload = _hourly_payload("2025-01-01T00:00", periods=24)
        responses = [
            _mk_response(503),
            _mk_response(503),
            _mk_response(200, payload),
        ]
        with patch.object(wf_module.requests, "get", side_effect=responses) as mock_get:
            df = fetcher.fetch_weather_data("2025-01-01", "2025-01-01", max_retries=3, initial_backoff=0.01)

        assert mock_get.call_count == 3
        assert len(df) == 24

    def test_retry_exhausted_raises(self, fetcher):
        # Keep returning 503 forever.
        with patch.object(wf_module.requests, "get", return_value=_mk_response(503)) as mock_get:
            with pytest.raises(requests.exceptions.HTTPError):
                fetcher.fetch_weather_data("2025-01-01", "2025-01-01", max_retries=2, initial_backoff=0.01)

        # max_retries=2 -> total 3 attempts.
        assert mock_get.call_count == 3

    @pytest.mark.xfail(
        reason=(
            "Production retries on every HTTPError, including 4xx auth failures (401/403). "
            "An auth failure will never be fixed by retrying, but the current code calls the API "
            "max_retries+1 times anyway. Expected: exactly 1 call for 401/403."
        ),
        strict=True,
    )
    def test_no_retry_on_auth_failure(self, fetcher):
        with patch.object(wf_module.requests, "get", return_value=_mk_response(401)) as mock_get:
            with pytest.raises(requests.exceptions.HTTPError):
                fetcher.fetch_weather_data("2025-01-01", "2025-01-01", max_retries=3, initial_backoff=0.01)
        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# 3. Forecast vs archive selection + 4. days_ahead clamp


class TestForecastVsArchive:
    def test_forecast_endpoint_used_for_future(self, fetcher):
        # Build a forecast payload spanning the requested window.
        future_start = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
        future_end = future_start + pd.Timedelta(hours=23)
        payload = _hourly_payload(future_start.strftime("%Y-%m-%dT%H:00"), periods=48)

        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, payload)) as mock_get:
            fetcher.fetch_forecast_data(future_start, future_end)

        called_url = mock_get.call_args.args[0]
        assert called_url == WeatherFetcher.FORECAST_BASE_URL
        params = mock_get.call_args.kwargs["params"]
        assert "forecast_days" in params
        assert "start_date" not in params

    def test_days_ahead_clamped_to_16(self, fetcher):
        # Ask for 30 days into the future; clamp at L130-133 should cap at 16.
        future_start = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
        future_end = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)
        payload = _hourly_payload(future_start.strftime("%Y-%m-%dT%H:00"), periods=24 * 17)

        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, payload)) as mock_get:
            fetcher.fetch_forecast_data(future_start, future_end)

        params = mock_get.call_args.kwargs["params"]
        # Behavioural lock: the silent clamp keeps days_ahead <= 16.
        assert params["forecast_days"] == 16

    def test_days_ahead_clamped_to_minimum_1(self, fetcher):
        # Range entirely in the past produces days_ahead<=0, which the clamp lifts to 1.
        past_start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
        past_end = past_start + pd.Timedelta(hours=23)
        payload = _hourly_payload(past_start.strftime("%Y-%m-%dT%H:00"), periods=24)

        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, payload)) as mock_get:
            fetcher.fetch_forecast_data(past_start, past_end)

        params = mock_get.call_args.kwargs["params"]
        assert params["forecast_days"] >= 1


# ---------------------------------------------------------------------------
# 5. Hybrid mode (past + future window)


class TestHybridMode:
    def test_splits_into_archive_and_forecast_calls(self, fetcher):
        # Window: 10 days ago -> 5 days ahead. Should hit both APIs.
        now = pd.Timestamp.now(tz="UTC")
        start = now - pd.Timedelta(days=10)
        end = now + pd.Timedelta(days=5)

        archive_payload = _hourly_payload(start.strftime("%Y-%m-%dT%H:00"), periods=24 * 6)
        forecast_payload = _hourly_payload((now - pd.Timedelta(days=5)).strftime("%Y-%m-%dT%H:00"), periods=24 * 11)

        def _side_effect(url, params=None, timeout=None):
            if url == WeatherFetcher.ARCHIVE_BASE_URL:
                return _mk_response(200, archive_payload)
            if url == WeatherFetcher.FORECAST_BASE_URL:
                return _mk_response(200, forecast_payload)
            raise AssertionError(f"unexpected URL: {url}")

        with patch.object(wf_module.requests, "get", side_effect=_side_effect) as mock_get:
            df = fetcher.fetch_weather_data_hybrid(start, end)

        urls_called = [c.args[0] for c in mock_get.call_args_list]
        assert WeatherFetcher.ARCHIVE_BASE_URL in urls_called
        assert WeatherFetcher.FORECAST_BASE_URL in urls_called
        # Exactly two calls (one archive, one forecast).
        assert len(urls_called) == 2
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# 6. tz normalization


class TestTzNormalization:
    def test_naive_input_yields_utc_index(self, fetcher):
        payload = _hourly_payload("2025-01-01T00:00", periods=24)
        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, payload)):
            df = fetcher.fetch_weather_data("2025-01-01", "2025-01-01")

        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"


# ---------------------------------------------------------------------------
# 7. & 8. Cache hit / cache-miss + gap fetch


class TestCacheBehaviour:
    def test_cache_hit_skips_api(self, fetcher, tmp_path: Path):
        # Pre-populate cache covering the entire requested range.
        idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {col: [1.0] * len(idx) for col in WeatherFetcher.HOURLY_PARAMS},
            index=idx,
        )
        cache_path = tmp_path / "weather_cache.parquet"
        WeatherFetcher.save_to_cache(cache_path, cached)

        with patch.object(wf_module.requests, "get") as mock_get:
            df = fetcher.get_weather_data(
                start="2025-01-01T05:00",
                end="2025-01-01T15:00",
                cache_path=cache_path,
                timezone="UTC",
            )

        # No HTTP call at all.
        assert mock_get.call_count == 0
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11  # 05:00..15:00 inclusive
        # Values match what we pre-populated.
        assert (df["temperature_2m"] == 1.0).all()

    def test_cache_miss_fetches_gap_and_stitches(self, fetcher, tmp_path: Path):
        # Cache covers a 10-hour window in the past (well below the 5-day archive cutoff).
        cache_start = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=30)
        idx = pd.date_range(cache_start, periods=10, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {col: [42.0] * len(idx) for col in WeatherFetcher.HOURLY_PARAMS},
            index=idx,
        )
        cache_path = tmp_path / "weather_cache.parquet"
        WeatherFetcher.save_to_cache(cache_path, cached)

        # Request a range that extends 5 hours past the cache end -> gap fetch.
        req_start = cache_start
        req_end = cache_start + pd.Timedelta(hours=14)

        # Gap payload: 5 hours after the cache end.
        gap_start = cache_start + pd.Timedelta(hours=10)
        gap_payload = _hourly_payload(gap_start.strftime("%Y-%m-%dT%H:00"), periods=24)

        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, gap_payload)) as mock_get:
            df = fetcher.get_weather_data(
                start=req_start,
                end=req_end,
                cache_path=cache_path,
                timezone="UTC",
            )

        # Exactly one HTTP call for the gap.
        assert mock_get.call_count == 1
        # Result stitches: cache portion (42.0) + new portion (not 42.0).
        assert len(df) >= 15  # 10 from cache + 5 from gap (inclusive)
        # Cache half retained its sentinel value.
        cache_half = df.loc[df.index < gap_start]
        assert (cache_half["temperature_2m"] == 42.0).all()
        # Gap half came from the API payload (values start near 10.0, not 42.0).
        gap_half = df.loc[df.index >= gap_start]
        assert (gap_half["temperature_2m"] != 42.0).any()

    def test_load_save_round_trip(self, tmp_path: Path):
        idx = pd.date_range("2025-06-01", periods=12, freq="h", tz="UTC")
        original = pd.DataFrame(
            {col: list(range(12)) for col in WeatherFetcher.HOURLY_PARAMS},
            index=idx,
        )
        original.index.name = "datetime"

        cache_path = tmp_path / "rt.parquet"
        WeatherFetcher.save_to_cache(cache_path, original)
        loaded = WeatherFetcher.load_from_cache(cache_path)

        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert loaded.index.tz is not None
        assert len(loaded) == 12
        pd.testing.assert_frame_equal(
            loaded.sort_index(axis=1),
            original.sort_index(axis=1),
            check_freq=False,
        )

    def test_load_from_cache_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            WeatherFetcher.load_from_cache(tmp_path / "does_not_exist.parquet")


# ---------------------------------------------------------------------------
# 9. Silent 24h cyclic fallback (the audit finding)


class TestSilentFallback:
    """The audit flagged: when API fails, ``_create_fallback_data`` silently
    repeats the last 24 hours of cached data without flagging the synthetic
    rows. This pollutes downstream training data. We DOCUMENT the behaviour
    here with an xfail so the failing assertion becomes the regression
    canary once production starts flagging fallback rows.
    """

    def test_fallback_rows_should_be_flagged_but_are_not(self, fetcher, tmp_path: Path):
        # Pre-populate cache with 30 hours of real data, then request a window
        # whose gap extends 5 hours past the cache end. Force the gap fetch
        # to fail so the silent fallback kicks in.
        cache_start = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=30)
        idx = pd.date_range(cache_start, periods=30, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {col: [7.0] * len(idx) for col in WeatherFetcher.HOURLY_PARAMS},
            index=idx,
        )
        cache_path = tmp_path / "weather_cache.parquet"
        WeatherFetcher.save_to_cache(cache_path, cached)

        req_start = cache_start
        req_end = cache_start + pd.Timedelta(hours=34)

        # Every API call fails.
        with patch.object(wf_module.requests, "get", return_value=_mk_response(503)):
            df = fetcher.get_weather_data(
                start=req_start,
                end=req_end,
                cache_path=cache_path,
                timezone="UTC",
                max_retries=1,
                fallback_on_failure=True,
            )

        # Sanity: it returned *something* rather than raising — that's the bug.
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 30  # cache + cyclic fallback rows

        # Audit expectation (currently UNMET): fallback rows should be flagged
        # via a column / multi-index / boolean. Today they aren't.
        synthetic_indicators = [
            "is_synthetic",
            "is_fallback",
            "synthetic",
            "fallback",
            "data_source",
            "source",
        ]
        present = [c for c in synthetic_indicators if c in df.columns]
        assert present, (
            "AUDIT FINDING: silent 24h cyclic fallback returns synthetic rows "
            "with NO indicator column. Downstream models cannot distinguish real "
            "weather from a repeated 24-hour pattern. Production should add a "
            "boolean flag column."
        )

    # Pin the xfail so we get a clear signal if production ever fixes it.
    test_fallback_rows_should_be_flagged_but_are_not = pytest.mark.xfail(
        reason=(
            "AUDIT FINDING (intentionally failing): _create_fallback_data silently "
            "synthesises a 24-hour cyclic pattern when API calls fail, with no "
            "indicator column. Downstream training cannot tell real from synthetic. "
            "Remove the xfail once production flags fallback rows."
        ),
        strict=True,
    )(test_fallback_rows_should_be_flagged_but_are_not)

    def test_fallback_returns_data_for_full_range(self, fetcher, tmp_path: Path):
        """Companion to the xfail: lock in the *current* (silent) behaviour
        so future refactors notice if the fallback stops firing entirely."""
        cache_start = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=30)
        idx = pd.date_range(cache_start, periods=30, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {col: [7.0] * len(idx) for col in WeatherFetcher.HOURLY_PARAMS},
            index=idx,
        )
        cache_path = tmp_path / "weather_cache.parquet"
        WeatherFetcher.save_to_cache(cache_path, cached)

        req_start = cache_start
        req_end = cache_start + pd.Timedelta(hours=34)

        with patch.object(wf_module.requests, "get", return_value=_mk_response(503)):
            df = fetcher.get_weather_data(
                start=req_start,
                end=req_end,
                cache_path=cache_path,
                timezone="UTC",
                max_retries=1,
                fallback_on_failure=True,
            )

        # The silent fallback fires -> we get rows for the post-cache gap too.
        assert len(df) > 30


# ---------------------------------------------------------------------------
# Additional scenarios


class TestEdgeCases:
    def test_empty_hourly_payload_raises(self, fetcher):
        # 200 OK but no ``hourly`` key -> domain raises ValueError.
        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, {})):
            with pytest.raises(ValueError, match="No hourly data"):
                fetcher.fetch_weather_data("2025-01-01", "2025-01-01")

    def test_api_error_in_payload_raises(self, fetcher):
        payload = {"error": True, "reason": "Invalid coordinates"}
        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, payload)):
            with pytest.raises(ValueError, match="Open-Meteo"):
                fetcher.fetch_weather_data("2025-01-01", "2025-01-01")

    def test_invalid_api_type_raises(self, fetcher):
        with pytest.raises(ValueError, match="api_type"):
            fetcher._fetch_from_api("2025-01-01", "2025-01-01", api_type="invalid")

    def test_missing_lat_lon_raises(self):
        with pytest.raises(ValueError, match="latitude and longitude"):
            WeatherFetcher({"latitude": 50.0})  # no longitude
        with pytest.raises(ValueError, match="latitude and longitude"):
            WeatherFetcher({"longitude": 7.0})  # no latitude

    def test_no_cache_path_fetches_from_api(self, fetcher):
        """Cache-less path delegates straight to fetch_weather_data_hybrid."""
        # Use a fully-historical range to keep things on the archive endpoint.
        cache_start = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=30)
        payload = _hourly_payload(cache_start.strftime("%Y-%m-%dT%H:00"), periods=48)
        with patch.object(wf_module.requests, "get", return_value=_mk_response(200, payload)) as mock_get:
            df = fetcher.get_weather_data(
                start=cache_start,
                end=cache_start + pd.Timedelta(hours=23),
                cache_path=None,
            )
        assert mock_get.call_count >= 1
        assert isinstance(df, pd.DataFrame)

    def test_run_is_thin_wrapper(self, fetcher, tmp_path: Path):
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        cached = pd.DataFrame(
            {col: [3.0] * len(idx) for col in WeatherFetcher.HOURLY_PARAMS},
            index=idx,
        )
        cache_path = tmp_path / "wc.parquet"
        WeatherFetcher.save_to_cache(cache_path, cached)
        # No HTTP needed because cache covers the range.
        with patch.object(wf_module.requests, "get") as mock_get:
            df = fetcher.run(
                start="2025-01-01T00:00",
                end="2025-01-01T05:00",
                cache_path=cache_path,
            )
        assert mock_get.call_count == 0
        assert len(df) == 6
