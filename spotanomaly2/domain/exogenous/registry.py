# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dynamic resolver that loads ExogenousFetcher/Joiner classes from YAML dotted paths."""

import importlib
from typing import Any, Iterator

from spotanomaly2.domain.exogenous.base import ExogenousFetcher, ExogenousJoiner


def _load_class(spec: str) -> type:
    """Resolve a ``'module.path:ClassName'`` spec to the actual class."""
    if ":" not in spec:
        raise ValueError(f"Class spec must be 'module:ClassName', got: {spec!r}")
    module_name, class_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def iter_configured_sources(
    config: dict[str, Any],
) -> Iterator[tuple[str, dict, type[ExogenousFetcher], type[ExogenousJoiner]]]:
    """Yield ``(name, source_cfg, FetcherCls, JoinerCls)`` for every entry in ``config['exogenous']``.

    A malformed entry (missing name/fetcher/joiner, unresolvable class) is skipped
    silently — orchestrators record the failure via their own ``fetch_status``
    accounting, not here.
    """
    for entry in config.get("exogenous", []) or []:
        name = entry.get("name")
        fetcher_spec = entry.get("fetcher")
        joiner_spec = entry.get("joiner")
        if not (name and fetcher_spec and joiner_spec):
            continue
        try:
            fetcher_cls = _load_class(fetcher_spec)
            joiner_cls = _load_class(joiner_spec)
        except (ImportError, AttributeError, ValueError):
            continue
        # Inject the YAML name so joiners can prefix columns as
        # ``exogenous_<name>_<measurement>`` — the convention the splitter,
        # detector, and report generator rely on to map columns back to a source.
        src_cfg = {**(entry.get("config") or {}), "source_name": name}
        yield name, src_cfg, fetcher_cls, joiner_cls
