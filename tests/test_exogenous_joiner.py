# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ExogenousJoiner — the process-stage orchestrator.

Mirror of test_exogenous_downloader but exercises the join path. Both
orchestrators share the registry-iteration pattern; tests here pin the
joiner-specific behaviour:
- panel_data flows through unchanged when no sources are configured,
- enabled sources' joiners receive panel_data and their return value chains,
- exceptions are caught and recorded as "error" without breaking the chain,
- disabled sources record status="disabled".
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from spotanomaly2.application.exogenous_joiner import ExogenousJoiner


@pytest.fixture
def panel_data():
    idx = pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC")
    return {"1": pd.DataFrame({"sensor": [1.0] * 3}, index=idx)}


def _install_fake_module(monkeypatch, module_name: str, **classes):
    mod = types.ModuleType(module_name)
    for name, cls in classes.items():
        setattr(mod, name, cls)
    monkeypatch.setitem(sys.modules, module_name, mod)


def _fetcher_class(*, enabled: bool = True):
    class _F:
        @classmethod
        def is_enabled(cls, source_config):
            return enabled

        def __init__(self, source_config, parent_config, logger=None):
            pass

        def fetch_and_cache(self, start, end):
            pass

    return _F


class TestEmptyConfig:
    def test_no_sources_returns_panel_data_unchanged(self, panel_data):
        joiner = ExogenousJoiner({"exogenous": []})
        assert joiner.join_all(panel_data) is panel_data


class TestDisabledSource:
    def test_disabled_records_status_and_skips_construction(self, monkeypatch, panel_data):
        joiner_calls: list = []

        class _J:
            def __init__(self, source_config, parent_config, logger=None):
                pass

            def join_into_panels(self, panel_data):
                joiner_calls.append(panel_data)
                return panel_data

        _install_fake_module(monkeypatch, "tests._fake_j_off", F=_fetcher_class(enabled=False), J=_J)
        cfg = {
            "exogenous": [
                {
                    "name": "off",
                    "fetcher": "tests._fake_j_off:F",
                    "joiner": "tests._fake_j_off:J",
                    "config": {},
                }
            ],
        }
        status: dict = {}
        ExogenousJoiner(cfg).join_all(panel_data, status)

        assert status["off"]["status"] == "disabled"
        assert joiner_calls == []


class TestEnabledSource:
    def test_join_receives_panel_data_and_chains_return_value(self, monkeypatch, panel_data):
        class _J:
            def __init__(self, source_config, parent_config, logger=None):
                pass

            def join_into_panels(self, pd_in):
                return {pid: df.assign(extra=42) for pid, df in pd_in.items()}

        _install_fake_module(monkeypatch, "tests._fake_j_ok", F=_fetcher_class(), J=_J)
        cfg = {
            "exogenous": [
                {
                    "name": "ok",
                    "fetcher": "tests._fake_j_ok:F",
                    "joiner": "tests._fake_j_ok:J",
                    "config": {},
                }
            ],
        }
        status: dict = {}
        result = ExogenousJoiner(cfg).join_all(panel_data, status)

        assert all("extra" in df.columns for df in result.values())
        assert status["ok"]["status"] == "ok"
        assert status["ok"]["error"] is None


class TestComposition:
    def test_multiple_joiners_chain_in_order(self, monkeypatch, panel_data):
        class _Ja:
            def __init__(self, source_config, parent_config, logger=None):
                pass

            def join_into_panels(self, pd_in):
                return {pid: df.assign(a=1) for pid, df in pd_in.items()}

        class _Jb:
            def __init__(self, source_config, parent_config, logger=None):
                pass

            def join_into_panels(self, pd_in):
                return {pid: df.assign(b=df["a"] + 1) for pid, df in pd_in.items()}

        _install_fake_module(monkeypatch, "tests._fake_j_chain", F=_fetcher_class(), Ja=_Ja, Jb=_Jb)
        cfg = {
            "exogenous": [
                {
                    "name": "first",
                    "fetcher": "tests._fake_j_chain:F",
                    "joiner": "tests._fake_j_chain:Ja",
                    "config": {},
                },
                {
                    "name": "second",
                    "fetcher": "tests._fake_j_chain:F",
                    "joiner": "tests._fake_j_chain:Jb",
                    "config": {},
                },
            ],
        }
        status: dict = {}
        result = ExogenousJoiner(cfg).join_all(panel_data, status)

        # Second saw first's output (column 'a' present → 'b' computed from it).
        assert all(df["b"].iloc[0] == 2 for df in result.values())
        assert status["first"]["status"] == "ok"
        assert status["second"]["status"] == "ok"


class TestFailurePath:
    def test_join_exception_records_error_and_continues(self, monkeypatch, panel_data):
        class _Jboom:
            def __init__(self, source_config, parent_config, logger=None):
                pass

            def join_into_panels(self, pd_in):
                raise RuntimeError("disk read failed")

        _install_fake_module(monkeypatch, "tests._fake_j_boom", F=_fetcher_class(), J=_Jboom)
        cfg = {
            "exogenous": [
                {
                    "name": "boom",
                    "fetcher": "tests._fake_j_boom:F",
                    "joiner": "tests._fake_j_boom:J",
                    "config": {},
                }
            ],
        }
        status: dict = {}
        result = ExogenousJoiner(cfg).join_all(panel_data, status)

        # On failure: chain breaks cleanly, panel_data flows through.
        assert result is panel_data
        assert status["boom"]["status"] == "error"
        assert "disk read failed" in status["boom"]["error"]
