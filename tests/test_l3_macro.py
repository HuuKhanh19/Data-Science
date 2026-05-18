"""
Tests for ``src.features.l3_macro``.

Strategy
--------
- Tier 2 golden values: hand-computed expected values for each of the 5
  features on synthetic inputs.
- Tier 1 anti-leak: 2 global mutation tests (daily features; asof-join
  features). The asof_join primitive itself is verified in test_asof_join.py
  (Session 2, 100% coverage) — this file only checks the composition.
- Calendar alignment (D3, D4): explicit tests for the FX-gap ffill path
  and the VN-Index strict-reindex path.
- Composer (compute_l3_features): integration test on synthetic full
  pipeline producing 5-column output.

All synthetic; no I/O against ``data/raw/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.l3_macro import (
    L3_FEATURE_COLS,
    compute_l3_features,
    compute_usdvnd_change,
    compute_vnindex_return,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bdates(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, name="date")


@pytest.fixture
def tcb_index_30day() -> pd.DatetimeIndex:
    return _bdates(30)


@pytest.fixture
def price_df_30day(tcb_index_30day) -> pd.DataFrame:
    return pd.DataFrame(
        {"adj_close": np.linspace(50.0, 60.0, 30)}, index=tcb_index_30day
    )


@pytest.fixture
def vnindex_df_30day(tcb_index_30day) -> pd.DataFrame:
    """VN-Index aligned to same calendar as TCB."""
    return pd.DataFrame(
        {"adj_close": np.linspace(1000.0, 1100.0, 30)}, index=tcb_index_30day
    )


@pytest.fixture
def fx_df_30day(tcb_index_30day) -> pd.DataFrame:
    """USD/VND on TCB calendar (no gaps; FX-gap tested separately)."""
    return pd.DataFrame(
        {"rate": np.linspace(23000.0, 24000.0, 30)}, index=tcb_index_30day
    )


@pytest.fixture
def macro_monthly_df() -> pd.DataFrame:
    """3 monthly releases covering 2024-01..2024-03 with valid schema."""
    return pd.DataFrame(
        {
            "reference_period": ["2024-01", "2024-02", "2024-03"],
            "release_date": pd.to_datetime(
                ["2024-02-14", "2024-03-14", "2024-04-14"]
            ),
            "release_date_source": ["scraped", "scraped", "fallback_14d"],
            "cpi_yoy_pct": [3.5, 3.6, 3.7],
            "sbv_refinancing_rate_pct": [4.5, 4.5, 4.0],
        }
    )


@pytest.fixture
def macro_quarterly_df() -> pd.DataFrame:
    """2 quarterly releases (Q4-2023, Q1-2024)."""
    return pd.DataFrame(
        {
            "reference_quarter": ["2023-Q4", "2024-Q1"],
            "release_date": pd.to_datetime(["2024-01-31", "2024-04-30"]),
            "release_date_source": ["scraped", "scraped"],
            "gdp_yoy_pct": [5.0, 5.6],
        }
    )


# ===========================================================================
# 1. compute_vnindex_return — golden + strict alignment
# ===========================================================================


def test_vnindex_return_golden(tcb_index_30day, vnindex_df_30day):
    """At t=2: log(I_1 / I_0). VN-Index linspace(1000, 1100, 30)."""
    result = compute_vnindex_return(vnindex_df_30day, tcb_index_30day)

    i0 = vnindex_df_30day["adj_close"].iloc[0]
    i1 = vnindex_df_30day["adj_close"].iloc[1]
    assert result.iloc[2] == pytest.approx(np.log(i1 / i0))


def test_vnindex_return_first_two_rows_nan(tcb_index_30day, vnindex_df_30day):
    """Need I_{t-2} ⇒ first 2 rows NaN."""
    result = compute_vnindex_return(vnindex_df_30day, tcb_index_30day)
    assert result.iloc[:2].isna().all()
    assert not result.iloc[2:].isna().any()


def test_vnindex_return_raises_on_missing_dates(tcb_index_30day):
    """If TCB index has a date that VN-Index lacks, raise (D4 strict)."""
    # VN-Index missing day 5
    short_index = tcb_index_30day.delete(5)
    vnindex_df = pd.DataFrame(
        {"adj_close": np.linspace(1000, 1100, len(short_index))},
        index=short_index,
    )
    with pytest.raises(ValueError, match="VN-Index is missing"):
        compute_vnindex_return(vnindex_df, tcb_index_30day)


def test_vnindex_return_anti_leak(tcb_index_30day, vnindex_df_30day):
    """Mutating last VN-Index value must not change the last feature row."""
    baseline = compute_vnindex_return(vnindex_df_30day, tcb_index_30day)
    mutated_df = vnindex_df_30day.copy()
    mutated_df.iloc[-1, mutated_df.columns.get_loc("adj_close")] = 99999.0
    mutated = compute_vnindex_return(mutated_df, tcb_index_30day)

    assert mutated.iloc[-1] == pytest.approx(baseline.iloc[-1])


# ===========================================================================
# 2. compute_usdvnd_change — golden + FX-gap ffill
# ===========================================================================


def test_usdvnd_change_golden(tcb_index_30day, fx_df_30day):
    """At t=2: log(X_1 / X_0)."""
    result = compute_usdvnd_change(fx_df_30day, tcb_index_30day)
    x0 = fx_df_30day["rate"].iloc[0]
    x1 = fx_df_30day["rate"].iloc[1]
    assert result.iloc[2] == pytest.approx(np.log(x1 / x0))


def test_usdvnd_change_handles_fx_gap_via_ffill(tcb_index_30day):
    """
    FX calendar missing some TCB trading days ⇒ ffill ⇒ 0 return on gap day.

    Setup: FX has values only on days 0, 1, 3 (missing day 2). On day 3
    of TCB, the feature is log(X_2 / X_1) where X_2 is forward-filled
    from X_1 ⇒ log(X_1/X_1) = 0.
    """
    fx_dates = tcb_index_30day[[0, 1, 3]]  # missing index 2
    fx_df = pd.DataFrame({"rate": [23000.0, 23100.0, 23200.0]}, index=fx_dates)

    result = compute_usdvnd_change(fx_df, tcb_index_30day)

    # Day 2 (FX-gap day): feature[2] = log(X_1/X_0) where X is ffilled.
    # After ffill, X[0]=23000, X[1]=23100, X[2]=23100 (filled), X[3]=23200.
    # feature[2] = log(X[1]/X[0]) = log(23100/23000).
    assert result.iloc[2] == pytest.approx(np.log(23100 / 23000))
    # Day 3: feature[3] = log(X[2]/X[1]) = log(23100/23100) = 0.
    assert result.iloc[3] == pytest.approx(0.0, abs=1e-12)
    # Day 4: feature[4] = log(X[3]/X[2]) = log(23200/23100).
    assert result.iloc[4] == pytest.approx(np.log(23200 / 23100))


def test_usdvnd_change_dates_before_fx_start_remain_nan(tcb_index_30day):
    """If FX series starts after TCB index, early TCB rows ⇒ NaN."""
    # FX starts at TCB day 10
    fx_dates = tcb_index_30day[10:]
    fx_df = pd.DataFrame(
        {"rate": np.linspace(23000, 24000, len(fx_dates))}, index=fx_dates
    )
    result = compute_usdvnd_change(fx_df, tcb_index_30day)
    # Days 0..10 have no prior FX values → NaN
    assert result.iloc[:11].isna().all()


def test_usdvnd_change_anti_leak(tcb_index_30day, fx_df_30day):
    baseline = compute_usdvnd_change(fx_df_30day, tcb_index_30day)
    mutated_df = fx_df_30day.copy()
    mutated_df.iloc[-1, mutated_df.columns.get_loc("rate")] = 99999.0
    mutated = compute_usdvnd_change(mutated_df, tcb_index_30day)
    assert mutated.iloc[-1] == pytest.approx(baseline.iloc[-1])


# ===========================================================================
# 3. compute_l3_features — composer (asof-join features integrated)
# ===========================================================================


def _wider_setup():
    """Wider fixtures so asof-join features have data to forward-fill from."""
    # TCB calendar spans 2024-01 through 2024-06 (~125 business days)
    tcb_idx = pd.bdate_range("2024-01-02", "2024-06-30", name="date")
    price_df = pd.DataFrame(
        {"adj_close": np.linspace(50.0, 60.0, len(tcb_idx))}, index=tcb_idx
    )
    vnindex_df = pd.DataFrame(
        {"adj_close": np.linspace(1000.0, 1100.0, len(tcb_idx))}, index=tcb_idx
    )
    fx_df = pd.DataFrame(
        {"rate": np.linspace(23000.0, 24000.0, len(tcb_idx))}, index=tcb_idx
    )
    macro_m = pd.DataFrame(
        {
            "reference_period": ["2024-01", "2024-02", "2024-03"],
            "release_date": pd.to_datetime(
                ["2024-02-14", "2024-03-14", "2024-04-14"]
            ),
            "release_date_source": ["scraped"] * 3,
            "cpi_yoy_pct": [3.5, 3.6, 3.7],
            "sbv_refinancing_rate_pct": [4.5, 4.5, 4.0],
        }
    )
    macro_q = pd.DataFrame(
        {
            "reference_quarter": ["2023-Q4", "2024-Q1"],
            "release_date": pd.to_datetime(["2024-01-31", "2024-04-30"]),
            "release_date_source": ["scraped"] * 2,
            "gdp_yoy_pct": [5.0, 5.6],
        }
    )
    return tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q


def test_compute_l3_features_returns_expected_columns():
    tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    result = compute_l3_features(price_df, vnindex_df, fx_df, macro_m, macro_q)
    assert tuple(result.columns) == L3_FEATURE_COLS


def test_compute_l3_features_preserves_index():
    tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    result = compute_l3_features(price_df, vnindex_df, fx_df, macro_m, macro_q)
    assert result.index.equals(tcb_idx)


def test_compute_l3_features_cpi_forward_filled_correctly():
    """
    CPI value at trading day t = value with largest release_date ≤ t.

    Macro_monthly rows:
        2024-02-14: cpi=3.5
        2024-03-14: cpi=3.6
        2024-04-14: cpi=3.7

    Expected:
        Before 2024-02-14: NaN (no release yet)
        2024-02-14 .. 2024-03-13: 3.5
        2024-03-14 .. 2024-04-13: 3.6
        2024-04-14 onwards:       3.7
    """
    tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    result = compute_l3_features(price_df, vnindex_df, fx_df, macro_m, macro_q)

    # Pick representative business days
    assert pd.isna(result.loc[pd.Timestamp("2024-02-13"), "cpi_yoy_pct"])
    assert result.loc[pd.Timestamp("2024-02-14"), "cpi_yoy_pct"] == 3.5
    assert result.loc[pd.Timestamp("2024-03-13"), "cpi_yoy_pct"] == 3.5
    assert result.loc[pd.Timestamp("2024-03-14"), "cpi_yoy_pct"] == 3.6
    assert result.loc[pd.Timestamp("2024-04-15"), "cpi_yoy_pct"] == 3.7
    assert result.loc[pd.Timestamp("2024-06-28"), "cpi_yoy_pct"] == 3.7


def test_compute_l3_features_sbv_forward_filled_correctly():
    """Same as CPI but for SBV refinancing rate."""
    tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    result = compute_l3_features(price_df, vnindex_df, fx_df, macro_m, macro_q)

    assert result.loc[pd.Timestamp("2024-02-14"), "sbv_rate_pct"] == 4.5
    assert result.loc[pd.Timestamp("2024-03-15"), "sbv_rate_pct"] == 4.5
    assert result.loc[pd.Timestamp("2024-04-15"), "sbv_rate_pct"] == 4.0


def test_compute_l3_features_gdp_forward_filled_correctly():
    """GDP releases at 2024-01-31 (Q4-2023) and 2024-04-30 (Q1-2024)."""
    tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    result = compute_l3_features(price_df, vnindex_df, fx_df, macro_m, macro_q)

    # Before 2024-01-31: NaN
    assert pd.isna(result.loc[pd.Timestamp("2024-01-30"), "gdp_yoy_pct"])
    assert result.loc[pd.Timestamp("2024-01-31"), "gdp_yoy_pct"] == 5.0
    assert result.loc[pd.Timestamp("2024-04-29"), "gdp_yoy_pct"] == 5.0
    assert result.loc[pd.Timestamp("2024-04-30"), "gdp_yoy_pct"] == 5.6
    assert result.loc[pd.Timestamp("2024-06-28"), "gdp_yoy_pct"] == 5.6


# ===========================================================================
# 4. Anti-leak Tier 1 — global
# ===========================================================================


def test_anti_leak_mutating_last_daily_inputs_does_not_affect_last_row():
    """Mutate last VN-Index AND last FX values; last feature row unchanged."""
    tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    baseline = compute_l3_features(price_df, vnindex_df, fx_df, macro_m, macro_q)

    vnindex_mut = vnindex_df.copy()
    vnindex_mut.iloc[-1, vnindex_mut.columns.get_loc("adj_close")] = 99999.0
    fx_mut = fx_df.copy()
    fx_mut.iloc[-1, fx_mut.columns.get_loc("rate")] = 99999.0

    mutated = compute_l3_features(price_df, vnindex_mut, fx_mut, macro_m, macro_q)
    # Daily features at last row must be unchanged
    assert mutated["vnindex_return"].iloc[-1] == pytest.approx(
        baseline["vnindex_return"].iloc[-1]
    )
    assert mutated["usdvnd_change"].iloc[-1] == pytest.approx(
        baseline["usdvnd_change"].iloc[-1]
    )


def test_anti_leak_future_macro_release_not_visible_before_release_date():
    """
    A macro release dated 2024-05-15 must NOT appear in features at any
    trading day before 2024-05-15. Add a fake future release and verify
    that features before its release_date are unchanged from baseline.
    """
    tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    baseline = compute_l3_features(price_df, vnindex_df, fx_df, macro_m, macro_q)

    # Inject a future release with a wildly different value
    future_row = pd.DataFrame(
        {
            "reference_period": ["2024-04"],
            "release_date": pd.to_datetime(["2024-05-15"]),
            "release_date_source": ["scraped"],
            "cpi_yoy_pct": [99.9],
            "sbv_refinancing_rate_pct": [99.9],
        }
    )
    macro_m_with_future = pd.concat([macro_m, future_row], ignore_index=True)

    with_future = compute_l3_features(
        price_df, vnindex_df, fx_df, macro_m_with_future, macro_q
    )

    # On 2024-05-14 (one day before the future release): CPI/SBV unchanged
    d = pd.Timestamp("2024-05-14")
    assert with_future.loc[d, "cpi_yoy_pct"] == baseline.loc[d, "cpi_yoy_pct"]
    assert with_future.loc[d, "sbv_rate_pct"] == baseline.loc[d, "sbv_rate_pct"]
    # On 2024-05-15: future release should now be visible
    d2 = pd.Timestamp("2024-05-15")
    assert with_future.loc[d2, "cpi_yoy_pct"] == 99.9


# ===========================================================================
# 5. Input validation
# ===========================================================================


def test_raises_on_non_datetimeindex_price_df():
    bad_price = pd.DataFrame({"adj_close": [1.0, 2.0]}, index=[0, 1])
    _, _, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_l3_features(bad_price, vnindex_df, fx_df, macro_m, macro_q)


def test_raises_on_vnindex_missing_adj_close():
    tcb_idx, price_df, _, fx_df, macro_m, macro_q = _wider_setup()
    bad_vn = pd.DataFrame({"close": [1.0] * len(tcb_idx)}, index=tcb_idx)
    with pytest.raises(ValueError, match="adj_close"):
        compute_l3_features(price_df, bad_vn, fx_df, macro_m, macro_q)


def test_raises_on_fx_missing_rate():
    tcb_idx, price_df, vnindex_df, _, macro_m, macro_q = _wider_setup()
    bad_fx = pd.DataFrame({"price": [1.0] * len(tcb_idx)}, index=tcb_idx)
    with pytest.raises(ValueError, match="rate"):
        compute_l3_features(price_df, vnindex_df, bad_fx, macro_m, macro_q)


def test_raises_on_macro_monthly_missing_column():
    tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    bad_macro = macro_m.drop(columns=["cpi_yoy_pct"])
    with pytest.raises(ValueError, match="cpi_yoy_pct"):
        compute_l3_features(price_df, vnindex_df, fx_df, bad_macro, macro_q)


def test_raises_on_macro_quarterly_missing_column():
    tcb_idx, price_df, vnindex_df, fx_df, macro_m, macro_q = _wider_setup()
    bad_q = macro_q.drop(columns=["gdp_yoy_pct"])
    with pytest.raises(ValueError, match="gdp_yoy_pct"):
        compute_l3_features(price_df, vnindex_df, fx_df, macro_m, bad_q)