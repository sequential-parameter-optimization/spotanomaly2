# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exogenous data sources: fetchers + joiners that enrich panel data.

YAML wires each source to a fetcher/joiner class pair by dotted path —
third-party packages can ship their own implementations and be plugged in
without patching spotanomaly2.
"""

from spotanomaly2.domain.exogenous.base import ExogenousFetcher, ExogenousJoiner

__all__ = ["ExogenousFetcher", "ExogenousJoiner"]
