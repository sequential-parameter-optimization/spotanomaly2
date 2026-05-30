# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ExogenousDownloader — the download-stage orchestrator.

These tests pin the per-source iteration contract:
- disabled sources record status="disabled" and skip construction,
- enabled sources are instantiated, fetched, and record status="ok",
- exceptions during construction or fetch record status="error",
- malformed YAML entries (missing name/fetcher/joiner) are silently skipped,
- ``fetch_status=None`` is a no-op.

The Fetcher/Joiner classes themselves are exercised by their own test files.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from spotanomaly2.application.exogenous_downloader import ExogenousDownloader


@pytest.fixture
def time_window():
    return (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-02", tz="UTC"),
    )


def _install_fake_module(monkeypatch, module_name: str, **classes):
    """Install a synthetic Python module so iter_configured_sources can import it."""
    mod = types.ModuleType(module_name)
    for name, cls in classes.items():
        setattr(mod, name, cls)
    monkeypatch.setitem(sys.modules, module_name, mod)


def _ok_fetcher_class(*, instances: list, calls: list, enabled: bool = True):
    class _OkFetcher:
        @classmethod
        def is_enabled(cls, source_config):
            return enabled

        def __init__(self, source_config, parent_config, logger=None):
            instances.append(self)
            self.source_config = source_config

        def fetch_and_cache(self, start, end, ignore_cache=False):
            calls.append((start, end))

    return _OkFetcher


def _noop_joiner_class():
    class _NoopJoiner:
        def __init__(self, source_config, parent_config, logger=None):
            pass

        def join_into_panels(self, panel_data):
            return panel_data

    return _NoopJoiner


class TestDisabledSource:
    def test_disabled_source_records_status_and_skips_construction(self, monkeypatch, time_window):
        instances: list = []
        calls: list = []
        FetcherCls = _ok_fetcher_class(instances=instances, calls=calls, enabled=False)
        JoinerCls = _noop_joiner_class()
        _install_fake_module(monkeypatch, "tests._fake_dl_off", F=FetcherCls, J=JoinerCls)

        cfg = {
            "exogenous": [
                {
                    "name": "off",
                    "fetcher": "tests._fake_dl_off:F",
                    "joiner": "tests._fake_dl_off:J",
                    "config": {},
                }
            ],
        }
        downloader = ExogenousDownloader(cfg)
        status: dict = {}
        downloader.download_all(*time_window, status)

        assert status["off"]["status"] == "disabled"
        assert instances == []
        assert calls == []


class TestEnabledSource:
    def test_fetch_and_cache_invoked_with_window_and_status_ok(self, monkeypatch, time_window):
        instances: list = []
        calls: list = []
        FetcherCls = _ok_fetcher_class(instances=instances, calls=calls)
        JoinerCls = _noop_joiner_class()
        _install_fake_module(monkeypatch, "tests._fake_dl_ok", F=FetcherCls, J=JoinerCls)

        cfg = {
            "exogenous": [
                {
                    "name": "ok_src",
                    "fetcher": "tests._fake_dl_ok:F",
                    "joiner": "tests._fake_dl_ok:J",
                    "config": {"foo": "bar"},
                }
            ],
        }
        downloader = ExogenousDownloader(cfg)
        status: dict = {}
        downloader.download_all(*time_window, status)

        assert len(instances) == 1
        assert calls == [time_window]
        assert status["ok_src"]["status"] == "ok"
        assert status["ok_src"]["error"] is None


class TestFailurePaths:
    def test_fetch_exception_records_error(self, monkeypatch, time_window):
        class _BoomFetcher:
            @classmethod
            def is_enabled(cls, source_config):
                return True

            def __init__(self, source_config, parent_config, logger=None):
                pass

            def fetch_and_cache(self, start, end, ignore_cache=False):
                raise RuntimeError("network down")

        _install_fake_module(monkeypatch, "tests._fake_dl_boom", F=_BoomFetcher, J=_noop_joiner_class())

        cfg = {
            "exogenous": [
                {
                    "name": "boom",
                    "fetcher": "tests._fake_dl_boom:F",
                    "joiner": "tests._fake_dl_boom:J",
                    "config": {},
                }
            ],
        }
        downloader = ExogenousDownloader(cfg)
        status: dict = {}
        downloader.download_all(*time_window, status)

        assert status["boom"]["status"] == "error"
        assert "network down" in status["boom"]["error"]

    def test_construction_exception_records_error(self, monkeypatch, time_window):
        class _BadCtorFetcher:
            @classmethod
            def is_enabled(cls, source_config):
                return True

            def __init__(self, source_config, parent_config, logger=None):
                raise ValueError("missing credentials")

            def fetch_and_cache(self, start, end, ignore_cache=False):
                pass  # never reached

        _install_fake_module(monkeypatch, "tests._fake_dl_ctor", F=_BadCtorFetcher, J=_noop_joiner_class())

        cfg = {
            "exogenous": [
                {
                    "name": "bad",
                    "fetcher": "tests._fake_dl_ctor:F",
                    "joiner": "tests._fake_dl_ctor:J",
                    "config": {},
                }
            ],
        }
        downloader = ExogenousDownloader(cfg)
        status: dict = {}
        downloader.download_all(*time_window, status)

        assert status["bad"]["status"] == "error"
        assert "missing credentials" in status["bad"]["error"]


class TestMalformedEntries:
    def test_entry_missing_fetcher_silently_skipped(self, time_window):
        cfg = {
            "exogenous": [
                {"name": "no_fetcher", "joiner": "spotanomaly2.x:Y", "config": {}},
            ],
        }
        downloader = ExogenousDownloader(cfg)
        status: dict = {}
        downloader.download_all(*time_window, status)
        assert status == {}

    def test_unresolvable_dotted_path_silently_skipped(self, time_window):
        cfg = {
            "exogenous": [
                {
                    "name": "ghost",
                    "fetcher": "nonexistent.module:Ghost",
                    "joiner": "nonexistent.module:Ghost",
                    "config": {},
                }
            ],
        }
        downloader = ExogenousDownloader(cfg)
        status: dict = {}
        downloader.download_all(*time_window, status)
        assert status == {}


class TestFetchStatusOptional:
    def test_none_status_does_not_raise(self, monkeypatch, time_window):
        instances: list = []
        calls: list = []
        FetcherCls = _ok_fetcher_class(instances=instances, calls=calls)
        JoinerCls = _noop_joiner_class()
        _install_fake_module(monkeypatch, "tests._fake_dl_nostat", F=FetcherCls, J=JoinerCls)

        cfg = {
            "exogenous": [
                {
                    "name": "ok",
                    "fetcher": "tests._fake_dl_nostat:F",
                    "joiner": "tests._fake_dl_nostat:J",
                    "config": {},
                }
            ],
        }
        ExogenousDownloader(cfg).download_all(*time_window, fetch_status=None)
        assert calls == [time_window]
