"""
As-of join: anti-leakage merge of lower-frequency data into a daily panel.

CRITICAL semantics (IMPLEMENTATION.md §5.1, research_design.md §4.3 / §4.4)
---------------------------------------------------------------------------
For each trading day ``t`` in the daily panel, attach values from the row of
the lower-frequency frame whose ``release_date`` is the **largest value
``≤ t``**. This guarantees that no feature at day ``t`` depends on
information that was not yet publicly known on day ``t``.

The forward-fill is keyed by ``release_date``, **never** by
``reference_period`` (quarter end or month end). Using ``reference_period``
would constitute look-ahead leakage because reports are publicly disclosed
weeks to months after the period they describe. This function takes
``release_date`` as input as-given; it cannot detect a caller who passes
``reference_period_end`` instead. Test 6 in ``tests/test_asof_join.py``
demonstrates and guards against that mistake at the function level.

Public API
----------
``asof_join``
    Generic core. Works for any lower-frequency frame with a release_date
    column.
``asof_join_quarterly``, ``asof_join_monthly``
    Thin wrappers around ``asof_join`` for call-site clarity. They do not
    add validation; the operation is identical for both frequencies.

Implementation
--------------
Backed by :func:`pandas.merge_asof` with ``direction="backward"`` and
``allow_exact_matches=True``, which exactly implements the
"largest release_date ≤ t" semantic. ``backward`` means right key ≤ left
key; ``allow_exact_matches=True`` means equality is a match (so a release
on its own publication day is considered available that same day).
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def asof_join(
    daily_df: pd.DataFrame,
    low_freq_df: pd.DataFrame,
    value_cols: list[str],
    release_date_col: str = "release_date",
) -> pd.DataFrame:
    """
    As-of join lower-frequency data into a daily DataFrame.

    For each row at date ``t`` in ``daily_df``, attach values from the row
    of ``low_freq_df`` with the largest ``release_date ≤ t``. Forward-fill
    is implicit: at any ``t`` past the latest release, the latest available
    value is used (correct behaviour for production inference).

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily panel. Index must be a unique, monotonically ascending
        :class:`pandas.DatetimeIndex`.
    low_freq_df : pd.DataFrame
        Lower-frequency frame (monthly or quarterly). Must contain the
        column named by ``release_date_col`` and every column listed in
        ``value_cols``. Must not contain duplicate ``release_date`` values.
        Empty frames are allowed and yield all-NaN ``value_cols``.
    value_cols : list[str]
        Columns from ``low_freq_df`` to attach to ``daily_df``. Must be
        non-empty.
    release_date_col : str, default ``"release_date"``
        Name of the column in ``low_freq_df`` holding the release date.

        .. warning::
           This **must** be the publication date (when the figure became
           public knowledge), not a ``reference_period_end``,
           ``reference_quarter`` end, or similar period-end marker. Passing
           a period-end column here is a silent look-ahead leak. The
           function cannot detect this mistake — the caller is responsible.

    Returns
    -------
    pd.DataFrame
        Copy of ``daily_df`` with ``value_cols`` merged in. The
        ``release_date_col`` itself is **dropped** from the output to keep
        the column from being accidentally consumed as a feature
        downstream. Index and all original columns of ``daily_df`` are
        preserved verbatim. Rows at date ``t`` earlier than the earliest
        ``release_date`` in ``low_freq_df`` receive NaN in the merged
        columns.

    Raises
    ------
    TypeError
        If ``daily_df.index`` is not a :class:`pandas.DatetimeIndex`.
    ValueError
        Various preconditions; see the inline validators for messages.
        Notable cases: empty ``value_cols``; missing columns in
        ``low_freq_df``; NaN/NaT in the release_date column; duplicate
        release_dates; column-name conflicts between ``daily_df`` and the
        right-side keys.

    See Also
    --------
    pandas.merge_asof : Underlying primitive.
    asof_join_quarterly, asof_join_monthly : Frequency-named wrappers.

    Notes
    -----
    Memory: the function copies ``low_freq_df`` (subset + sort) and the
    output of ``merge_asof``; ``daily_df`` itself is not mutated. The
    output is a fresh ``DataFrame``.
    """
    _validate_daily(daily_df)
    _validate_low_freq(low_freq_df, value_cols, release_date_col)
    _validate_no_column_conflict(daily_df, value_cols, release_date_col)

    # Resolve the index column name we will use after reset_index.
    # If daily_df.index is unnamed, pandas reset_index produces a column
    # called 'index'; we rename it to a stable 'date' for clarity.
    index_name = daily_df.index.name if daily_df.index.name else "date"

    # Prepare left side: expose index as a column for merge_asof (which
    # operates on columns, not the index).
    left = daily_df.reset_index()
    if daily_df.index.name is None:
        left = left.rename(columns={"index": index_name})

    # Prepare right side: select only the columns we need, coerce
    # release_date to datetime, and sort. pd.merge_asof requires both
    # keys to be sorted ascending; we sort defensively even though the
    # validator already accepts any input order.
    right = low_freq_df[[release_date_col, *value_cols]].copy()
    right[release_date_col] = pd.to_datetime(right[release_date_col])
    right = right.sort_values(release_date_col).reset_index(drop=True)

    # Coerce the right join key to match the left's dtype. pd.merge_asof
    # in pandas 2.x requires *exact* dtype match on the join keys, but
    # different constructors yield different resolutions (e.g.
    # ``bdate_range`` → datetime64[us], ``pd.to_datetime(list_of_str)``
    # → datetime64[ns]). Matching right to left preserves daily_df's
    # original index dtype on the output, which downstream callers
    # (feature pipelines that pd.concat across multiple as-of joins)
    # rely on. For whole-day timestamps the astype is lossless in both
    # directions.
    target_dtype = left[index_name].dtype
    right[release_date_col] = right[release_date_col].astype(target_dtype)

    # Core merge. ``direction="backward"`` ⇒ right key ≤ left key;
    # ``allow_exact_matches=True`` ⇒ equality counts as a match (a release
    # on its own publication day is considered available that day).
    merged = pd.merge_asof(
        left=left,
        right=right,
        left_on=index_name,
        right_on=release_date_col,
        direction="backward",
        allow_exact_matches=True,
    )

    # Drop the release_date column from output. Decision D3 of Session 2:
    # exposing release_date downstream invites accidental use as a feature
    # (it carries weak time-of-year signal). Audit traceability is not
    # lost — callers retain ``low_freq_df`` itself.
    merged = merged.drop(columns=[release_date_col])

    # Restore the original index.
    merged = merged.set_index(index_name)
    merged.index.name = daily_df.index.name  # exact reproduction (incl. None)

    # Diagnostics.
    n_total = len(merged)
    nan_count_first = int(merged[value_cols[0]].isna().sum())
    log.info(
        "asof_join: daily=%d rows, low_freq=%d rows, value_cols=%s, "
        "NaN tail (rows before first release) in '%s' = %d",
        n_total,
        len(low_freq_df),
        value_cols,
        value_cols[0],
        nan_count_first,
    )

    return merged


def asof_join_quarterly(
    daily_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    value_cols: list[str],
    release_date_col: str = "release_date",
) -> pd.DataFrame:
    """
    Quarterly-named wrapper around :func:`asof_join`.

    Identical behaviour and parameter semantics; provided for call-site
    clarity (e.g. ``asof_join_quarterly(..., value_cols=["gdp_yoy_pct"])``
    documents intent better than the generic name). See :func:`asof_join`
    for full docstring.
    """
    return asof_join(daily_df, quarterly_df, value_cols, release_date_col)


def asof_join_monthly(
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    value_cols: list[str],
    release_date_col: str = "release_date",
) -> pd.DataFrame:
    """
    Monthly-named wrapper around :func:`asof_join`.

    Identical behaviour and parameter semantics; provided for call-site
    clarity. See :func:`asof_join` for full docstring.
    """
    return asof_join(daily_df, monthly_df, value_cols, release_date_col)


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------


def _validate_daily(daily_df: pd.DataFrame) -> None:
    """Enforce daily_df preconditions: DatetimeIndex, unique, sorted."""
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        raise TypeError(
            f"daily_df.index must be DatetimeIndex, got "
            f"{type(daily_df.index).__name__}"
        )
    if not daily_df.index.is_unique:
        dups = daily_df.index[daily_df.index.duplicated()].tolist()
        raise ValueError(
            f"daily_df.index has duplicate dates: "
            f"{dups[:5]}{'…' if len(dups) > 5 else ''}"
        )
    if not daily_df.index.is_monotonic_increasing:
        raise ValueError(
            "daily_df.index must be sorted ascending. Call "
            "daily_df.sort_index() before passing it in."
        )


def _validate_low_freq(
    low_freq_df: pd.DataFrame,
    value_cols: list[str],
    release_date_col: str,
) -> None:
    """Enforce low_freq_df preconditions."""
    if not value_cols:
        raise ValueError("value_cols must be a non-empty list of column names")

    if release_date_col not in low_freq_df.columns:
        raise ValueError(
            f"release_date_col '{release_date_col}' not in low_freq_df columns: "
            f"{low_freq_df.columns.tolist()}"
        )

    missing_value_cols = [c for c in value_cols if c not in low_freq_df.columns]
    if missing_value_cols:
        raise ValueError(
            f"value_cols missing from low_freq_df: {missing_value_cols}. "
            f"Available columns: {low_freq_df.columns.tolist()}"
        )

    # Empty frames are allowed (yield all-NaN merge); skip the rest of the
    # value-level checks if there is nothing to inspect.
    if len(low_freq_df) == 0:
        return

    release_coerced = pd.to_datetime(low_freq_df[release_date_col], errors="coerce")
    if release_coerced.isna().any():
        n_nan = int(release_coerced.isna().sum())
        raise ValueError(
            f"low_freq_df['{release_date_col}'] contains {n_nan} NaN/NaT or "
            "uncoercible values; release_date is required for every row"
        )

    if release_coerced.duplicated().any():
        dups = release_coerced[release_coerced.duplicated()].tolist()
        raise ValueError(
            f"low_freq_df['{release_date_col}'] has duplicate release_dates: "
            f"{dups[:5]}{'…' if len(dups) > 5 else ''}. "
            "Duplicate releases are ambiguous for as-of join; the caller must "
            "deduplicate explicitly (e.g. keep latest revision)."
        )


def _validate_no_column_conflict(
    daily_df: pd.DataFrame,
    value_cols: list[str],
    release_date_col: str,
) -> None:
    """Ensure no column-name collisions between left and right sides."""
    target_names = {release_date_col, *value_cols}

    col_conflicts = set(daily_df.columns) & target_names
    if col_conflicts:
        raise ValueError(
            f"daily_df has columns that would conflict with the merge: "
            f"{sorted(col_conflicts)}. Rename or drop these before calling "
            "asof_join (silent overwrite would mask data-prep errors)."
        )

    index_name = daily_df.index.name
    if index_name is not None and index_name in target_names:
        raise ValueError(
            f"daily_df.index.name '{index_name}' conflicts with "
            "release_date_col or a value_col. Rename the index before "
            "calling asof_join."
        )