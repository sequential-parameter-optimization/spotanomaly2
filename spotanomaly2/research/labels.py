# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ground-truth label assembly.

Combines two label sources:

1. ``known_anomalies`` from the project config (already used in
   ``spotforecast_adapter._mask_known_anomalies`` to *exclude* training
   data — here we reuse the timestamp ranges as *positive labels* for
   evaluation).
2. Synthetic injection labels produced by :mod:`spotanomaly2.research.synthetic`.

Output is always a binary :class:`pandas.Series` (int8, 0/1) aligned with
a caller-supplied DatetimeIndex.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def labels_from_known(
    index: pd.DatetimeIndex,
    known_anomalies: list[dict[str, Any]] | None,
    buffer: str = "0s",
) -> pd.Series:
    """Build a binary label vector from the config's ``known_anomalies`` list.

    Each entry is expected to be a dict with ``start`` and ``end`` ISO
    timestamps. Rows whose timestamp falls within ``[start - buffer,
    end + buffer]`` are labelled 1; everything else 0.

    Args:
        index: DatetimeIndex of the test set (the rows being scored).
        known_anomalies: List of {start, end} dicts from config, or None.
        buffer: Pandas timedelta string applied symmetrically around
            each window. Defaults to ``"0s"`` (no expansion). The training
            pipeline uses a 1-day buffer for masking; for evaluation we
            default to no buffer because labels should be tight.

    Returns:
        ``pd.Series`` of int8, indexed by ``index``.
    """
    labels = pd.Series(0, index=index, dtype=np.int8)
    if not known_anomalies:
        return labels

    buffer_td = pd.Timedelta(buffer) if buffer else pd.Timedelta(0)

    for anomaly in known_anomalies:
        start_str = anomaly.get("start")
        end_str = anomaly.get("end")
        if not start_str or not end_str:
            continue
        try:
            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
        except (ValueError, TypeError):
            continue

        if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
            if start.tz is None:
                start = start.tz_localize(index.tz)
            else:
                start = start.tz_convert(index.tz)
            if end.tz is None:
                end = end.tz_localize(index.tz)
            else:
                end = end.tz_convert(index.tz)

        mask = (index >= start - buffer_td) & (index <= end + buffer_td)
        labels.loc[mask] = 1

    return labels


def merge_labels(*sources: pd.Series) -> pd.Series:
    """Element-wise OR of multiple binary label Series with identical indices.

    Args:
        *sources: One or more aligned int/bool Series.

    Returns:
        int8 Series with 1 wherever any source has 1.

    Raises:
        ValueError: If sources have differing indices or no sources given.
    """
    if not sources:
        raise ValueError("merge_labels requires at least one source Series")

    base_index = sources[0].index
    for s in sources[1:]:
        if not s.index.equals(base_index):
            raise ValueError("All label sources must share the same index")

    merged = np.zeros(len(base_index), dtype=np.int8)
    for s in sources:
        merged = (merged | s.to_numpy().astype(np.int8)).astype(np.int8)
    return pd.Series(merged, index=base_index, dtype=np.int8)
