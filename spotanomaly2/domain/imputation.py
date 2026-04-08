"""Imputation utilities for filling missing values in time series data."""

import numpy as np
import pandas as pd


def identify_missing_data_gaps_with_count(df: pd.DataFrame | pd.Series) -> list[tuple]:
    """Identify gaps of missing data and count the number of missing points in each gap.

    Uses vectorized operations instead of iterrows for ~100x speedup on large
    DataFrames.

    Args:
        df: Pandas DataFrame or Series with missing values.

    Returns:
        List of ``(start_index, end_index, count)`` tuples, one per
        contiguous gap of missing data.
    """
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    is_missing = df.isna().any(axis=1)

    if not is_missing.any():
        return []

    # Detect boundaries: where the missing state changes
    shifted = is_missing.shift(1, fill_value=False)
    gap_starts = is_missing & ~shifted  # False->True transitions
    gap_ends = ~is_missing & shifted  # True->False transitions

    start_positions = np.where(gap_starts.values)[0]
    end_positions = np.where(gap_ends.values)[0]

    # If the series ends inside a gap, add the last index as end
    if len(start_positions) > len(end_positions):
        end_positions = np.append(end_positions, len(is_missing))

    gaps = []
    idx = df.index
    for s, e in zip(start_positions, end_positions):
        count = e - s
        gaps.append((idx[s], idx[e - 1], count))

    return gaps


def vectorized_subsequence_distances(series: pd.Series, subsequence: pd.Series) -> pd.DataFrame:
    """Compute Euclidean distance between `subsequence` and all windows in `series`."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a DatetimeIndex.")

    series_array = np.array(series).flatten()
    subsequence_array = np.array(subsequence).flatten()
    subsequence_len = len(subsequence)

    if subsequence_len > len(series_array):
        raise ValueError("Subsequence is longer than the series.")

    s_subsequences = np.lib.stride_tricks.sliding_window_view(series_array, subsequence_len)

    distances = np.linalg.norm(s_subsequences - subsequence_array, axis=1)

    df = pd.DataFrame(index=series.index[: len(series) - subsequence_len + 1])
    df["distance"] = distances
    df["start_idx"] = df.index
    df["end_idx"] = df.index + df.index.freq * (subsequence_len - 1)

    df.dropna(inplace=True)

    return df


def series_mean(one: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Element-wise mean between two arrays."""
    return (one + other) / 2


def fill_missing_with_mean(series: pd.Series) -> pd.Series:
    """Fill single missing values with the mean of their two neighbours.

    Only fills gaps of exactly length 1 (both neighbours must be non-NaN).
    Uses vectorized shift operations instead of a Python loop.

    Args:
        series: The series to fill missing values in.

    Returns:
        A copy of the series with single-value gaps filled.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("The parameter must be a Pandas Series.")

    result = series.copy()
    missing = result.isna()
    prev_valid = ~result.shift(1).isna()
    next_valid = ~result.shift(-1).isna()

    # Mask: NaN value with valid neighbours on both sides
    fillable = missing & prev_valid & next_valid
    if fillable.any():
        result[fillable] = (result.shift(1)[fillable] + result.shift(-1)[fillable]) / 2

    return result


def subsequence_imputation(
    series: pd.Series,
    distance_func=vectorized_subsequence_distances,
    weighting_func=series_mean,
    logger=None,
) -> pd.Series:
    """
    Impute missing values in a series using Partial Subsequence Matching (PSM).

    This implementation of the PSM algorithm first detects all gaps in the dataframe.
    For each gap of size n, subsequences of the same length of the gap are extracted, both on the left and right side.
    The distance between these two subsequences and every window of size n is computed using distance_func.
    The window with the smallest distance is selected for each side.
    The values of the left and right subsequences are then processed using weighting_func.
    The resulting values are then used to fill the gap.
    Currently, this imputation function does not work with single value gaps.
    Use fill_missing_with_mean before using this function to impute single value gaps.

    See https://doi.org/10.1007/s11269-022-03408-6 for more information.

    :param series: The pd.Series to impute missing data into.
    :param distance_func: The distance function to use for calculating distances between subsequences.
        Must return a DataFrame with columns 'start_idx', 'end_idx' and 'distance'. Defaults to vectorized function.
    :param weighting_func: The weighting function to use for combining the left and right subsequences.
        Must return a numpy array with the same length as the input arrays. Defaults to mean.
    :return: A copy of the given series with imputed values.
    """

    if not isinstance(series, pd.Series):
        raise ValueError("The parameter must be a Pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a DatetimeIndex.")

    if series.index.freq is None:
        raise ValueError("The series must have a frequency.")

    series = series.copy()
    freq = series.index.freq

    gaps = identify_missing_data_gaps_with_count(series)
    gaps = sorted(gaps, key=lambda x: x[2])

    for gap in gaps:
        start_idx = gap[0]
        end_idx = gap[1]
        missing_count = gap[2]

        l_start_idx = series.index.get_loc(start_idx) - missing_count
        r_end_idx = series.index.get_loc(end_idx) + missing_count

        l_s = None
        r_s = None

        if l_start_idx >= 0:
            l_subseq = series[l_start_idx : series.index.get_loc(start_idx)]
            distances = distance_func(series, l_subseq)
            sorted_distances = distances.sort_values(by="distance", ascending=True)

            for idx, distance in sorted_distances.iterrows():
                lower = (series.index.get_loc(distance["end_idx"])) + 1
                upper = (series.index.get_loc(distance["end_idx"])) + missing_count + 1
                l_s = series[lower:upper]

                if not l_s.isna().any().any() and (len(l_s.values) == missing_count):
                    break

        if r_end_idx + 1 <= len(series):
            r_subseq = series[series.index.get_loc(end_idx + freq) : r_end_idx + 1]
            distances = distance_func(series, r_subseq)
            sorted_distances = distances.sort_values(by="distance", ascending=True)

            for idx, distance in sorted_distances.iterrows():
                if (series.index.get_loc(distance["start_idx"])) - missing_count < 0:
                    continue
                lower = (series.index.get_loc(distance["start_idx"])) - missing_count
                upper = series.index.get_loc(distance["start_idx"])
                r_s = series[lower:upper]

                if not r_s.isna().any().any():
                    break

        if l_s is None and r_s is None:
            if logger:
                logger.warning(
                    f"PSM imputation: could not fill gap at "
                    f"{start_idx} -> {end_idx} ({missing_count} points) — "
                    f"no valid matching subsequence found on either side"
                )
            continue
        elif l_s is None:
            l_s = r_s
        elif r_s is None:
            r_s = l_s

        impute_values = weighting_func(l_s.values, r_s.values)
        series.loc[start_idx:end_idx] = impute_values
    return series
