"""
Tests for ``src.data.asof_join``.

Tier 1 critical anti-leakage suite (IMPLEMENTATION.md §5.1, §8). All tests
operate on synthetic data — no I/O against ``data/raw/``. Coverage target
for ``src.data.asof_join``: ≥70%.

Suite layout
------------
1.  Semantic tests (Tests 1–7): forward-fill, inclusivity at the boundary,
    behaviour between releases and after the final release, the critical
    "100-day lag" anti-leak proof, and multi-column joins.
2.  Input-validation tests (Tests 8–14): the function must refuse mal-formed
    inputs loudly rather than silently producing wrong results.
3.  Output-structure tests (Tests 15–17): the result must preserve the
    daily index and pre-existing columns, drop ``release_date``, and pass
    a property check against a manual oracle on 100 random sample dates.
4.  Wrapper tests (Tests 18–19): the two frequency-named wrappers must
    produce results bit-identical to the core function.

Fixtures are inline per the Session 2 decision (conftest.py deferred until
later sessions present concrete reuse opportunities).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.asof_join import (
    asof_join,
    asof_join_monthly,
    asof_join_quarterly,
)


# ---------------------------------------------------------------------------
# Inline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_df_2020() -> pd.DataFrame:
    """One business year of daily dates: 2020-01-01 through 2020-12-31.

    Index is named 'date' to match the convention from
    ``src/data/fetchers.py`` (Session 1). A trivial 'close' column is
    included so that we can assert column preservation downstream.
    """
    idx = pd.bdate_range("2020-01-01", "2020-12-31", name="date")
    return pd.DataFrame({"close": np.linspace(10.0, 20.0, len(idx))}, index=idx)


@pytest.fixture
def quarterly_df_2020() -> pd.DataFrame:
    """Four quarterly observations spanning Q4-2019 through Q3-2020.

    Release dates are placed ~45 days after each quarter end, matching
    the conservative convention in research_design.md §4.4 for the L4
    TCB fundamentals layer.

        reference_quarter  quarter_end   release_date
        2019-Q4            2019-12-31    2020-02-14
        2020-Q1            2020-03-31    2020-05-15
        2020-Q2            2020-06-30    2020-08-14
        2020-Q3            2020-09-30    2020-11-13

    Q3-2019 (which would have released ~Nov 2019) is intentionally
    omitted so that early days of 2020 hit the 'before first release'
    edge case and return NaN.

    Two value columns are provided to exercise multi-column joins.
    """
    return pd.DataFrame({
        "reference_quarter": ["2019-Q4", "2020-Q1", "2020-Q2", "2020-Q3"],
        "release_date": pd.to_datetime(
            ["2020-02-14", "2020-05-15", "2020-08-14", "2020-11-13"]
        ),
        "gdp_yoy_pct": [7.0, 3.8, 0.4, 2.6],
        "npl_pct": [1.2, 1.5, 1.8, 1.6],
    })


# ===========================================================================
# 1. Semantic tests
# ===========================================================================


def test_before_first_release_returns_nan(daily_df_2020, quarterly_df_2020):
    """Test 1 — at dates strictly before the earliest release, the merged
    column must be NaN. This is the boundary case at the lower end."""
    result = asof_join(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    before_first = result.loc[:"2020-02-13"]
    assert before_first["gdp_yoy_pct"].isna().all(), (
        "Rows before the first release date must have NaN — non-NaN values "
        "would indicate look-ahead leakage."
    )


def test_on_release_date_inclusive(daily_df_2020, quarterly_df_2020):
    """Test 2 — at ``t == release_date`` the row from that release is
    used. This is the ``≤`` semantic (inclusive of equality)."""
    result = asof_join(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    # 2020-02-14 was a Friday (a business day) and equals the first release.
    val = result.loc[pd.Timestamp("2020-02-14"), "gdp_yoy_pct"]
    assert val == 7.0


def test_just_after_release_date_uses_that_row(daily_df_2020, quarterly_df_2020):
    """Test 3 — the trading day immediately after a release still uses
    that release (no off-by-one in the backward direction)."""
    result = asof_join(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    # 2020-02-17 is the Monday following the release on Friday 2020-02-14.
    val = result.loc[pd.Timestamp("2020-02-17"), "gdp_yoy_pct"]
    assert val == 7.0


def test_between_releases_uses_older_release(daily_df_2020, quarterly_df_2020):
    """Test 4 — between two consecutive releases, the older release is
    in effect. Using the newer one would be look-ahead leakage."""
    result = asof_join(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    # 2020-07-15 lies between Q1 release (2020-05-15) and Q2 (2020-08-14).
    val = result.loc[pd.Timestamp("2020-07-15"), "gdp_yoy_pct"]
    assert val == 3.8, (
        "Between releases the older release must be used; got the newer "
        "release value, which would constitute look-ahead leakage."
    )


def test_after_latest_release_forward_fills(daily_df_2020, quarterly_df_2020):
    """Test 5 (Decision D4) — past the final release, the function
    forward-fills the latest known value indefinitely. This is the
    correct behaviour for production inference where 'today' is always
    past the most recent disclosure."""
    result = asof_join(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    # 2020-12-31 is well after the final release on 2020-11-13.
    val = result.loc[pd.Timestamp("2020-12-31"), "gdp_yoy_pct"]
    assert val == 2.6


def test_release_date_lag_100_days_does_not_use_reference_period():
    """Test 6 — **the critical anti-leak proof**.

    Synthetic data places the release_date exactly 100 days after the
    reference_period_end. If the implementation accidentally joined on
    reference_period_end, the value would 'leak in' 100 days early.
    This test refutes that failure mode at the function level.

    Construction
    ------------
    reference_period_end:   2020-03-31  (the period the figure describes)
    release_date:           2020-07-09  (the day it became public, 100 d later)
    metric value:           99.9

    Expected behaviour
    ------------------
    At t ∈ {2020-04-15, 2020-05-15, 2020-07-08} — all of which lie after
    the reference_period_end but before the release_date — the merged
    value must be **NaN**. A non-NaN value here is direct evidence of
    a leak.

    At t ∈ {2020-07-09 (release_date itself), 2020-07-10}, the merged
    value must be 99.9.
    """
    daily = pd.DataFrame(
        {"close": [1.0] * 5},
        index=pd.to_datetime(
            ["2020-04-15", "2020-05-15", "2020-07-08", "2020-07-09", "2020-07-10"]
        ),
    )
    daily.index.name = "date"

    low_freq = pd.DataFrame({
        "reference_period_end": pd.to_datetime(["2020-03-31"]),
        "release_date":         pd.to_datetime(["2020-07-09"]),
        "metric":               [99.9],
    })

    result = asof_join(daily, low_freq, value_cols=["metric"])

    # Pre-release: NaN (the proof).
    assert pd.isna(result.loc[pd.Timestamp("2020-04-15"), "metric"])
    assert pd.isna(result.loc[pd.Timestamp("2020-05-15"), "metric"])
    assert pd.isna(result.loc[pd.Timestamp("2020-07-08"), "metric"])
    # Release day and after: value present.
    assert result.loc[pd.Timestamp("2020-07-09"), "metric"] == 99.9
    assert result.loc[pd.Timestamp("2020-07-10"), "metric"] == 99.9


def test_multiple_value_cols_joined_together(daily_df_2020, quarterly_df_2020):
    """Test 7 — joining multiple ``value_cols`` in a single call must
    yield consistent values (same release row underlies all of them)."""
    result = asof_join(
        daily_df_2020,
        quarterly_df_2020,
        value_cols=["gdp_yoy_pct", "npl_pct"],
    )
    row = result.loc[pd.Timestamp("2020-07-15")]
    # Between Q1 (2020-05-15) and Q2 (2020-08-14) releases ⇒ Q1 row applies.
    assert row["gdp_yoy_pct"] == 3.8
    assert row["npl_pct"] == 1.5


# ===========================================================================
# 2. Input-validation tests
# ===========================================================================


def test_raises_on_missing_value_col(daily_df_2020, quarterly_df_2020):
    """Test 8 — a value_col absent from low_freq_df raises ValueError."""
    with pytest.raises(ValueError, match="value_cols missing"):
        asof_join(
            daily_df_2020, quarterly_df_2020, value_cols=["does_not_exist"]
        )


def test_raises_on_missing_release_date_col(daily_df_2020, quarterly_df_2020):
    """Test 9 — release_date column absent from low_freq_df raises."""
    df_no_release = quarterly_df_2020.drop(columns=["release_date"])
    with pytest.raises(ValueError, match="release_date_col"):
        asof_join(daily_df_2020, df_no_release, value_cols=["gdp_yoy_pct"])


def test_raises_on_non_datetimeindex_daily(quarterly_df_2020):
    """Test 10 — daily_df with non-DatetimeIndex raises TypeError."""
    daily = pd.DataFrame({"close": [1.0, 2.0]}, index=[0, 1])  # RangeIndex
    with pytest.raises(TypeError, match="DatetimeIndex"):
        asof_join(daily, quarterly_df_2020, value_cols=["gdp_yoy_pct"])


def test_raises_on_unsorted_daily_index(quarterly_df_2020):
    """Test 11 — unsorted DatetimeIndex raises rather than silently
    producing nonsense (merge_asof requires sorted keys)."""
    daily = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"]),
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        asof_join(daily, quarterly_df_2020, value_cols=["gdp_yoy_pct"])


def test_raises_on_duplicate_release_dates(daily_df_2020, quarterly_df_2020):
    """Test 12 (Decision D2) — duplicate release_dates raise ValueError
    rather than silently dedup-ing."""
    dup_row = pd.DataFrame({
        "reference_quarter": ["DUPE"],
        "release_date": pd.to_datetime(["2020-02-14"]),  # collides with Q4-2019
        "gdp_yoy_pct": [99.9],
        "npl_pct": [9.9],
    })
    dup_df = pd.concat([quarterly_df_2020, dup_row], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate release_dates"):
        asof_join(daily_df_2020, dup_df, value_cols=["gdp_yoy_pct"])


def test_raises_on_empty_value_cols(daily_df_2020, quarterly_df_2020):
    """Test 13 — empty value_cols raises (no plausible call site for
    this; cheap to guard)."""
    with pytest.raises(ValueError, match="non-empty"):
        asof_join(daily_df_2020, quarterly_df_2020, value_cols=[])


def test_raises_on_column_name_conflict(daily_df_2020, quarterly_df_2020):
    """Test 14 — if daily_df already contains a column with the same
    name as one of the value_cols, raise rather than silently overwrite."""
    daily_with_conflict = daily_df_2020.copy()
    daily_with_conflict["gdp_yoy_pct"] = 0.0  # collides with value_col
    with pytest.raises(ValueError, match="conflict"):
        asof_join(
            daily_with_conflict, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
        )


# ===========================================================================
# 3. Output-structure tests
# ===========================================================================


def test_preserves_daily_index_and_other_columns(daily_df_2020, quarterly_df_2020):
    """Test 15 — the daily_df index and any pre-existing columns must be
    preserved verbatim in the output.

    ``check_freq=False`` on the index/series comparisons because pandas
    drops the inferred frequency metadata (e.g. ``BusinessDay``) when
    a DatetimeIndex round-trips through ``reset_index → merge_asof →
    set_index``. The actual dates and values are preserved; only the
    cached ``freq`` attribute is lost. That is acceptable: callers do
    not depend on this metadata, and re-inferring it costs nothing if
    needed downstream.
    """
    result = asof_join(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    # Index.equals checks values + dtype but is lenient about the ``freq``
    # attribute; assert_index_equal in older pandas versions does not
    # accept a check_freq kwarg.
    assert result.index.equals(daily_df_2020.index)
    assert "close" in result.columns
    pd.testing.assert_series_equal(
        result["close"], daily_df_2020["close"], check_freq=False
    )


def test_release_date_dropped_from_output(daily_df_2020, quarterly_df_2020):
    """Test 16 (Decision D3) — the release_date column must NOT appear in
    the returned DataFrame."""
    result = asof_join(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    assert "release_date" not in result.columns


def test_empty_low_freq_returns_all_nan(daily_df_2020):
    """Test 17 — an empty low_freq_df is permitted and yields all-NaN
    value_cols with the daily index preserved."""
    empty = pd.DataFrame({
        "release_date": pd.Series([], dtype="datetime64[ns]"),
        "gdp_yoy_pct":  pd.Series([], dtype=float),
    })
    result = asof_join(daily_df_2020, empty, value_cols=["gdp_yoy_pct"])
    assert result["gdp_yoy_pct"].isna().all()
    assert result.index.equals(daily_df_2020.index)


def test_property_random_dates_match_manual_oracle(daily_df_2020, quarterly_df_2020):
    """Test 18 — property-style check against a manual oracle.

    For 100 random sample dates from ``daily_df_2020``, compute the
    expected value by hand (the value at the largest release_date ≤ t)
    and assert it matches the implementation. This catches edge cases
    that the per-case tests above could miss.
    """
    rng = np.random.default_rng(42)
    sampled_dates = rng.choice(
        daily_df_2020.index.to_numpy(), size=100, replace=False
    )

    result = asof_join(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    sorted_releases = quarterly_df_2020.sort_values("release_date").reset_index(
        drop=True
    )

    for t in sampled_dates:
        eligible = sorted_releases[sorted_releases["release_date"] <= t]
        expected = (
            eligible.iloc[-1]["gdp_yoy_pct"] if len(eligible) > 0 else np.nan
        )
        actual = result.loc[pd.Timestamp(t), "gdp_yoy_pct"]

        if pd.isna(expected):
            assert pd.isna(actual), f"At t={t}: expected NaN, got {actual}"
        else:
            assert actual == expected, (
                f"At t={t}: expected {expected}, got {actual}"
            )


# ===========================================================================
# 4. Wrapper tests
# ===========================================================================


def test_quarterly_wrapper_delegates_to_asof_join(daily_df_2020, quarterly_df_2020):
    """Test 19 — the quarterly-named wrapper produces results
    bit-identical to the core function."""
    core_out = asof_join(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    wrapper_out = asof_join_quarterly(
        daily_df_2020, quarterly_df_2020, value_cols=["gdp_yoy_pct"]
    )
    pd.testing.assert_frame_equal(core_out, wrapper_out)


def test_monthly_wrapper_delegates_to_asof_join(daily_df_2020):
    """Test 20 — the monthly-named wrapper produces results bit-identical
    to the core function. Uses a small monthly fixture with release_dates
    set to month-end + ~30 days, matching the CPI/SBV convention from
    research_design.md §4.3."""
    monthly = pd.DataFrame({
        "reference_period": ["2020-01", "2020-02", "2020-03"],
        "release_date":     pd.to_datetime(
            ["2020-02-28", "2020-03-31", "2020-04-30"]
        ),
        "cpi_yoy_pct":      [5.0, 4.5, 4.0],
    })
    core_out = asof_join(
        daily_df_2020, monthly, value_cols=["cpi_yoy_pct"]
    )
    wrapper_out = asof_join_monthly(
        daily_df_2020, monthly, value_cols=["cpi_yoy_pct"]
    )
    pd.testing.assert_frame_equal(core_out, wrapper_out)


# ===========================================================================
# 5. Defensive validator coverage (additional error paths)
# ===========================================================================


def test_handles_unnamed_daily_index(quarterly_df_2020):
    """Test 21 — daily_df with an unnamed DatetimeIndex still works
    (default fallback name 'date' is used internally)."""
    idx = pd.bdate_range("2020-01-01", "2020-06-30")  # no name= kwarg
    daily = pd.DataFrame({"close": [1.0] * len(idx)}, index=idx)
    result = asof_join(daily, quarterly_df_2020, value_cols=["gdp_yoy_pct"])
    assert "gdp_yoy_pct" in result.columns
    assert result.loc[pd.Timestamp("2020-05-15"), "gdp_yoy_pct"] == 3.8


def test_raises_on_duplicate_daily_dates(quarterly_df_2020):
    """Test 22 — daily_df with duplicate dates raises (would be ambiguous
    for the merge result)."""
    daily = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.to_datetime(["2020-06-01", "2020-06-01"]),
    )
    with pytest.raises(ValueError, match="duplicate dates"):
        asof_join(daily, quarterly_df_2020, value_cols=["gdp_yoy_pct"])


def test_raises_on_nan_release_date(daily_df_2020):
    """Test 23 — NaT in release_date column raises (each release must
    have a known publication date for as-of join to be meaningful)."""
    bad = pd.DataFrame({
        "release_date": pd.to_datetime(["2020-05-15", None]),
        "gdp_yoy_pct":  [3.8, 0.4],
    })
    with pytest.raises(ValueError, match="NaN/NaT"):
        asof_join(daily_df_2020, bad, value_cols=["gdp_yoy_pct"])


def test_raises_on_index_name_conflict(quarterly_df_2020):
    """Test 24 — daily_df with index.name colliding with a value_col
    raises (would create ambiguity after reset_index)."""
    idx = pd.bdate_range("2020-01-01", "2020-06-30", name="gdp_yoy_pct")
    daily = pd.DataFrame({"close": [1.0] * len(idx)}, index=idx)
    with pytest.raises(ValueError, match="index.name"):
        asof_join(daily, quarterly_df_2020, value_cols=["gdp_yoy_pct"])