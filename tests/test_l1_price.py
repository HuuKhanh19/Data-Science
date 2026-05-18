"""
Tests for ``src.features.l1_price``.

Tier 2 (golden hand-computed values) + Tier 1 (anti-leak) per
IMPLEMENTATION.md §8. All synthetic data, no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.l1_price import (
    L1_FEATURE_COLS,
    compute_l1_features,
)


# ---------------------------------------------------------------------------
# Fixtures (inline; conftest.py deferred until Session 4+ reuse emerges)
# ---------------------------------------------------------------------------


@pytest.fixture
def price_df_30day() -> pd.DataFrame:
    """30 business days, hand-picked prices (strictly positive, varied)."""
    idx = pd.bdate_range("2024-01-01", periods=30, name="date")
    prices = [
        10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.7, 11.9, 11.6,  # 0–9
        12.0, 12.3, 12.5, 12.2, 12.6, 12.8, 12.5, 13.0, 13.2, 12.9,  # 10–19
        13.3, 13.6, 13.8, 13.5, 13.9, 14.2, 14.0, 14.4, 14.6, 14.3,  # 20–29
    ]
    return pd.DataFrame({"adj_close": prices}, index=idx)


# ---------------------------------------------------------------------------
# 1. Golden value tests (Tier 2)
# ---------------------------------------------------------------------------


def test_r_lag1_golden_at_index_2(price_df_30day):
    """r_lag1[t=2] = log(P_1 / P_0) = log(10.5 / 10.0)."""
    result = compute_l1_features(price_df_30day)
    assert result["r_lag1"].iloc[2] == pytest.approx(np.log(10.5 / 10.0))


def test_r_lag1_golden_at_index_5(price_df_30day):
    """r_lag1[t=5] = log(P_4 / P_3) = log(11.2 / 10.8)."""
    result = compute_l1_features(price_df_30day)
    assert result["r_lag1"].iloc[5] == pytest.approx(np.log(11.2 / 10.8))


def test_r_cum5_golden(price_df_30day):
    """r_cum5[t=6] = log(P_5 / P_0) = log(11.5 / 10.0). (Span of 5 days.)"""
    result = compute_l1_features(price_df_30day)
    assert result["r_cum5"].iloc[6] == pytest.approx(np.log(11.5 / 10.0))


def test_r_cum10_golden(price_df_30day):
    """r_cum10[t=11] = log(P_10 / P_0) = log(12.0 / 10.0). (Span of 10.)"""
    result = compute_l1_features(price_df_30day)
    assert result["r_cum10"].iloc[11] == pytest.approx(np.log(12.0 / 10.0))


def test_r_cum20_golden(price_df_30day):
    """r_cum20[t=21] = log(P_20 / P_0) = log(13.3 / 10.0). (Span of 20.)"""
    result = compute_l1_features(price_df_30day)
    assert result["r_cum20"].iloc[21] == pytest.approx(np.log(13.3 / 10.0))


# ---------------------------------------------------------------------------
# 2. Warmup NaN pattern
# ---------------------------------------------------------------------------


def test_warmup_nan_pattern_per_feature(price_df_30day):
    """Each feature has expected number of leading NaN rows."""
    result = compute_l1_features(price_df_30day)
    # r_lag1 needs P_{t-2}, so rows 0,1 NaN
    assert result["r_lag1"].iloc[:2].isna().all()
    assert not result["r_lag1"].iloc[2:].isna().any()
    # r_cum5 needs P_{t-6}, so rows 0..5 NaN
    assert result["r_cum5"].iloc[:6].isna().all()
    assert not result["r_cum5"].iloc[6:].isna().any()
    # r_cum10 needs P_{t-11}, so rows 0..10 NaN
    assert result["r_cum10"].iloc[:11].isna().all()
    assert not result["r_cum10"].iloc[11:].isna().any()
    # r_cum20 needs P_{t-21}, so rows 0..20 NaN
    assert result["r_cum20"].iloc[:21].isna().all()
    assert not result["r_cum20"].iloc[21:].isna().any()


# ---------------------------------------------------------------------------
# 3. Anti-leak property tests (Tier 1)
# ---------------------------------------------------------------------------


def test_anti_leak_current_day_price_not_used(price_df_30day):
    """
    CRITICAL anti-leak proof. Mutating P_t (the current day's price)
    must NOT change any feature value at index t — features must depend
    only on P_{t-1} and earlier.

    We mutate the last row's price to a wildly different value and
    verify that compute_l1_features returns identical values for that
    row before and after.
    """
    original = compute_l1_features(price_df_30day)

    mutated_df = price_df_30day.copy()
    last_idx = mutated_df.index[-1]
    mutated_df.loc[last_idx, "adj_close"] = 999.0  # massive perturbation

    mutated = compute_l1_features(mutated_df)

    # Last row of features must be identical (uses only P_{t-1} and earlier)
    pd.testing.assert_series_equal(
        original.iloc[-1], mutated.iloc[-1], check_names=False
    )


def test_anti_leak_only_previous_prices_referenced():
    """
    Stronger anti-leak proof: replace ALL of P_t at every t > 21 with
    NaN, then verify features computed by compute_l1_features at row 21
    match the values computed from the prefix-only price series.

    If the function leaked future information, the all-NaN suffix would
    poison the row-21 output.
    """
    n = 50
    rng = np.random.default_rng(42)
    base_prices = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        index=pd.bdate_range("2024-01-01", periods=n, name="date"),
    )
    full_df = pd.DataFrame({"adj_close": base_prices})

    # Series with everything from row 22 onwards NaN
    nan_suffix_prices = base_prices.copy()
    nan_suffix_prices.iloc[22:] = np.nan
    nan_suffix_df = pd.DataFrame({"adj_close": nan_suffix_prices})

    full_result = compute_l1_features(full_df)
    suffix_result = compute_l1_features(nan_suffix_df)

    # Row 21 (and earlier) must match exactly: it only sees P_0..P_20
    # which are identical in both inputs.
    pd.testing.assert_frame_equal(
        full_result.iloc[: 22],
        suffix_result.iloc[: 22],
    )


# ---------------------------------------------------------------------------
# 4. Output shape and column contract
# ---------------------------------------------------------------------------


def test_output_has_expected_columns(price_df_30day):
    result = compute_l1_features(price_df_30day)
    assert tuple(result.columns) == L1_FEATURE_COLS


def test_output_preserves_index(price_df_30day):
    result = compute_l1_features(price_df_30day)
    assert result.index.equals(price_df_30day.index)


def test_custom_price_col_supported():
    """`price_col` param honoured for non-default column names."""
    idx = pd.bdate_range("2024-01-01", periods=25, name="date")
    df = pd.DataFrame({"close_raw": np.linspace(10.0, 20.0, 25)}, index=idx)
    result = compute_l1_features(df, price_col="close_raw")
    # Just check the function ran and produced expected shape
    assert tuple(result.columns) == L1_FEATURE_COLS
    assert len(result) == 25


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------


def test_raises_on_missing_price_col(price_df_30day):
    with pytest.raises(ValueError, match="price_col"):
        compute_l1_features(price_df_30day, price_col="does_not_exist")


def test_raises_on_non_positive_price():
    idx = pd.bdate_range("2024-01-01", periods=25, name="date")
    bad_prices = [10.0] * 24 + [0.0]
    df = pd.DataFrame({"adj_close": bad_prices}, index=idx)
    with pytest.raises(ValueError, match="non-positive"):
        compute_l1_features(df)


def test_raises_on_negative_price():
    idx = pd.bdate_range("2024-01-01", periods=25, name="date")
    bad_prices = [10.0] * 24 + [-1.5]
    df = pd.DataFrame({"adj_close": bad_prices}, index=idx)
    with pytest.raises(ValueError, match="non-positive"):
        compute_l1_features(df)


def test_allows_nan_in_input_propagates_to_output():
    """NaN in input price propagates to features that depend on it,
    but does not raise (NaN is an absence, not a violation)."""
    idx = pd.bdate_range("2024-01-01", periods=25, name="date")
    prices = list(np.linspace(10.0, 20.0, 25))
    prices[10] = np.nan
    df = pd.DataFrame({"adj_close": prices}, index=idx)
    result = compute_l1_features(df)
    # Row 11: r_lag1 = log(P_10/P_9) = log(NaN/...) = NaN
    assert pd.isna(result["r_lag1"].iloc[11])
    # Row 12: r_lag1 = log(P_11/P_10) = log(.../NaN) = NaN
    assert pd.isna(result["r_lag1"].iloc[12])
    # Row 13 onwards should be fine again (P_12 and P_11 are valid)
    assert not pd.isna(result["r_lag1"].iloc[13])