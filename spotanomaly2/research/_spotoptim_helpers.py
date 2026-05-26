# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Search-space conversion helpers for SpotOptim.

These two functions are duplicated from
``spotforecast2/src/spotforecast2/model_selection/spotoptim_search.py``
(``convert_search_space`` and ``array_to_params``). Duplication is preferred
over importing them because spotforecast2's surface around those helpers
is forecaster-specific (the surrounding ``spotoptim_search_forecaster``
expects forecaster objects), and we want a small stable surface for
scorer tuning that doesn't accidentally drift with forecaster changes.

If spotforecast2 ever exposes these as a stable public utility module we
should switch to importing them.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def convert_search_space(
    search_space: dict[str, Any],
) -> tuple[list[Any], list[str], list[str], list[Callable | None]]:
    """Translate a dict search-space into SpotOptim's bounds/types/names/transforms.

    Supported entry shapes (per name):

    - ``(low, high)`` — float or int bounds (int when both endpoints are int)
    - ``(low, high, "log10")`` — float bounds with log10 transform
    - ``[a, b, c, ...]`` — categorical (factor) — sampled by index

    Returns:
        Tuple ``(bounds, var_type, var_name, var_trans)`` matching
        SpotOptim constructor kwargs.
    """
    if not isinstance(search_space, dict):
        raise TypeError(f"search_space must be dict, got {type(search_space)}")

    bounds: list[Any] = []
    var_type: list[str] = []
    var_name: list[str] = []
    var_trans: list[Callable | str | None] = []

    for name, value in search_space.items():
        var_name.append(name)

        if (
            isinstance(value, tuple)
            and len(value) in (2, 3)
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            if isinstance(value[0], int) and isinstance(value[1], int):
                var_type.append("int")
            else:
                var_type.append("float")
            bounds.append(value[:2])
            var_trans.append(value[2] if len(value) == 3 else None)
        elif isinstance(value, list):
            var_type.append("factor")
            bounds.append(value)
            var_trans.append(None)
        else:
            raise ValueError(f"Invalid search-space entry for {name!r}: {value!r}")

    return bounds, var_type, var_name, var_trans


def array_to_params(
    params_array: np.ndarray,
    var_name: list[str],
    var_type: list[str],
    bounds: list[Any],
) -> dict[str, Any]:
    """Convert a SpotOptim 1-D parameter array back into a typed dict.

    Mirrors the production helper: ``"int"`` → ``int``, ``"float"`` →
    ``float``, ``"factor"`` → category lookup by string match or by
    rounded index fallback.
    """
    out: dict[str, Any] = {}
    for i, (name, ptype, value) in enumerate(zip(var_name, var_type, params_array)):
        if ptype == "factor":
            str_value = str(value)
            if str_value in bounds[i]:
                out[name] = str_value
            else:
                try:
                    idx = int(round(float(str_value)))
                    idx = max(0, min(idx, len(bounds[i]) - 1))
                    out[name] = bounds[i][idx]
                except (ValueError, TypeError):
                    out[name] = str_value
        elif ptype == "int":
            out[name] = int(round(float(str(value))))
        else:
            out[name] = float(str(value))
    return out
