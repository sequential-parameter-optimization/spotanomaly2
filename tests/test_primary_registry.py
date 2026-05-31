# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the pluggable primary-fetcher resolver."""

import pytest

from spotanomaly2.domain.primary_registry import resolve_primary_fetcher
from spotanomaly2.examples.entsoe_fetcher import EntsoePrimaryFetcher


def test_missing_fetcher_raises():
    # There is no implicit default: a config without `primary.fetcher` must fail
    # with a clear, actionable error.
    with pytest.raises(ValueError, match="primary.fetcher"):
        resolve_primary_fetcher({})
    with pytest.raises(ValueError, match="primary.fetcher"):
        resolve_primary_fetcher({"primary": {"display_name": "x"}})


def test_resolves_dotted_spec_to_configured_class():
    config = {
        "primary": {
            "fetcher": "spotanomaly2.examples.entsoe_fetcher:EntsoePrimaryFetcher",
            "config": {"country_code": "DE"},
        }
    }
    fetcher = resolve_primary_fetcher(config)
    assert isinstance(fetcher, EntsoePrimaryFetcher)
    assert fetcher.country_code == "DE"


def test_malformed_spec_raises():
    with pytest.raises(ValueError, match="module:ClassName"):
        resolve_primary_fetcher({"primary": {"fetcher": "no_colon_here"}})
