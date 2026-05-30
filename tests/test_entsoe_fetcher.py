# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for EntsoePrimaryFetcher (network mocked)."""

import entsoe
import pandas as pd
import pytest

from spotanomaly2.examples.entsoe_fetcher import EntsoePrimaryFetcher


class _FakeClient:
    """Stand-in for entsoe.EntsoePandasClient returning a tz-aware local load frame."""

    def __init__(self, api_key):
        self.api_key = api_key

    def query_load_and_forecast(self, country_code, start, end):
        idx = pd.date_range("2024-06-01", periods=48, freq="h", tz="Europe/Berlin")
        return pd.DataFrame(
            {"Forecasted Load": range(48), "Actual Load": range(100, 148)},
            index=idx,
        )


@pytest.fixture
def config(tmp_path):
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")  # no key here, so env controls the outcome
    return {
        "paths": {"credentials_file": str(empty_env)},
        "fetch": {"start_date": "2024-06-01T00:00:00+00:00", "end_date": "2024-06-03T00:00:00+00:00"},
        "primary": {"config": {"country_code": "DE", "channel_name": "load"}},
    }


def test_run_shapes_actual_load_into_panel(config, monkeypatch):
    monkeypatch.setenv("ENTSOE_API_KEY", "test-token")
    monkeypatch.setattr(entsoe, "EntsoePandasClient", _FakeClient)

    result = EntsoePrimaryFetcher(config).run()

    assert set(result) == {"DE"}
    df = result["DE"]
    assert list(df.columns) == ["channel_0_load"]
    assert df.index.name == "timestamp"
    assert str(pd.DatetimeIndex(df.index).tz) == "UTC"
    assert len(df) == 48
    # "Actual Load" (100..147) is selected, not "Forecasted Load".
    assert df["channel_0_load"].iloc[0] == 100


def test_missing_api_key_raises(config, monkeypatch):
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    monkeypatch.setattr(entsoe, "EntsoePandasClient", _FakeClient)

    with pytest.raises(ValueError, match="ENTSOE_API_KEY"):
        EntsoePrimaryFetcher(config).run()
