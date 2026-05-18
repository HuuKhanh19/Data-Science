"""
Tests for ``src.features.l2_technical``.

Strategy
--------
- Tier 2 (golden hand-computed values): one or more numerical examples
  per indicator, with intermediate steps annotated in the test body so
  an auditor can verify by hand.
- Tier 1 (anti-leak): mutating P_t (the current day's price) must NOT
  change any L2 feature value at index ``t`` — every indicator depends
  only on P_{t-1} and earlier.
- Edge cases (constant price, monotonic up/down, division-by-zero
  paths) are explicit tests where the indicator has a well-defined
  limit value.

All data synthetic; no I/O. Hand-computations in test comments cite the
formula in research_design.md §4.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.l2_technical import (
    L2_FEATURE_COLS,
    compute_bb_position,
    compute_l2_features,
    compute_ma_crossover,
    compute_macd_normalized,
    compute_momentum_12_3,
    compute_rsi_wilder,
    compute_trb_signal,
)


# ===========================================================================
# Helpers for tests
# ===========================================================================


def _bdates(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, name="date")


# ===========================================================================
# 1. MA crossover (MA_5 / MA_20)
# ===========================================================================


def test_ma_crossover_constant_price_yields_one():
    """Constant prices ⇒ MA_5 == MA_20 ⇒ ratio == 1 (after warmup)."""
    price = pd.Series([10.0] * 30, index=_bdates(30))
    result = compute_ma_crossover(price)
    # Warmup: 20 rows from MA_20 + 1 from shift = first 20 NaN, valid from 20
    assert result.iloc[:20].isna().all()
    assert result.iloc[20:].eq(1.0).all()


def test_ma_crossover_golden_index_20():
    """
    With price[0..14] = 10 and price[15..19] = 12, at index 20:
      MA_5  = mean(P_{t-5..t-1}) = mean(P_15..P_19) = mean([12]*5) = 12
      MA_20 = mean(P_{t-20..t-1}) = mean(P_0..P_19) = (15*10 + 5*12)/20 = 10.5
      ratio = 12 / 10.5 = 8/7 ≈ 1.142857
    """
    prices = [10.0] * 15 + [12.0] * 5 + [10.0] * 5
    price = pd.Series(prices, index=_bdates(25))
    result = compute_ma_crossover(price)
    assert result.iloc[20] == pytest.approx(12.0 / 10.5)


def test_ma_crossover_warmup_pattern():
    """20 leading NaN (need MA_20 + 1-step shift)."""
    price = pd.Series(np.linspace(10.0, 20.0, 25), index=_bdates(25))
    result = compute_ma_crossover(price)
    assert result.iloc[:20].isna().all()
    assert not result.iloc[20:].isna().any()


# ===========================================================================
# 2. Momentum 12-3 (decision D7 = B; skip-3 J-T convention)
# ===========================================================================


def test_momentum_12_3_golden_at_index_252():
    """
    With price[0..99]=10 and price[100..]=12, at t=252:
      mom = log(P_{t-63}/P_{t-252}) = log(P_189 / P_0) = log(12/10).
    Confirms the skip-3 convention: P_{t-63} (not P_{t-1}).
    """
    prices = [10.0] * 100 + [12.0] * 160
    price = pd.Series(prices, index=_bdates(260))
    result = compute_momentum_12_3(price)
    assert result.iloc[252] == pytest.approx(np.log(12.0 / 10.0))


def test_momentum_12_3_anti_leak_recent_63_days_dont_affect():
    """
    If momentum uses log(P_{t-63}/P_{t-252}), mutating any of
    P_{t-62..t-1} or P_t itself must NOT change momentum at t. This is
    the empirical proof that decision D7=B was implemented (under D7=A,
    P_{t-1} WOULD be used and the test would fail).
    """
    prices = [10.0] * 100 + [12.0] * 160
    price = pd.Series(prices, index=_bdates(260))
    baseline = compute_momentum_12_3(price)

    # Mutate the most recent 63 prices (positions 197..259, inclusive)
    mutated = price.copy()
    mutated.iloc[197:] = 99.0
    mutated_result = compute_momentum_12_3(mutated)

    # momentum at 252 should be IDENTICAL — only depends on P_189 and P_0.
    assert mutated_result.iloc[252] == pytest.approx(baseline.iloc[252])
    assert mutated_result.iloc[252] == pytest.approx(np.log(12.0 / 10.0))


def test_momentum_12_3_warmup_pattern():
    """First 252 rows NaN."""
    price = pd.Series(np.linspace(10.0, 20.0, 260), index=_bdates(260))
    result = compute_momentum_12_3(price)
    assert result.iloc[:252].isna().all()
    assert not result.iloc[252:].isna().any()


# ===========================================================================
# 3. Bollinger Band position
# ===========================================================================


def test_bb_position_golden_index_20():
    """
    price[0..18] = 10, price[19] = 12. At t=20:
      window = P_0..P_19 = [10]*19 + [12]
      MA_20 = (19*10 + 12)/20 = 202/20 = 10.1
      variance (ddof=0) = (19*(10 - 10.1)² + (12 - 10.1)²)/20
                        = (19*0.01 + 3.61)/20 = 3.80/20 = 0.19
      std_20 = √0.19 ≈ 0.4358899
      P_{t-1} = P_19 = 12
      BB_pos = (12 - 10.1) / (2 * 0.4358899) = 1.9 / 0.8717798 ≈ 2.17945
    """
    prices = [10.0] * 19 + [12.0] + [15.0]  # P_20 = 15 (won't affect BB_pos[20])
    price = pd.Series(prices, index=_bdates(21))
    result = compute_bb_position(price)
    expected = 1.9 / (2 * np.sqrt(0.19))
    assert result.iloc[20] == pytest.approx(expected, rel=1e-9)


def test_bb_position_constant_price_yields_nan():
    """Constant prices ⇒ std = 0 ⇒ BB_pos = (0)/(0) = NaN."""
    price = pd.Series([10.0] * 30, index=_bdates(30))
    result = compute_bb_position(price)
    # After warmup, std=0; (P_prev - MA) is also 0, so 0/0 = NaN
    assert result.iloc[20:].isna().all()


def test_bb_position_symmetric_below_mean_is_negative():
    """If P_{t-1} below the rolling mean, BB position is negative."""
    prices = [10.0] * 19 + [8.0] + [9.0]  # P_19 = 8 (below mean of 10)
    price = pd.Series(prices, index=_bdates(21))
    result = compute_bb_position(price)
    assert result.iloc[20] < 0


def test_bb_position_warmup_pattern():
    price = pd.Series(np.linspace(10.0, 20.0, 25), index=_bdates(25))
    result = compute_bb_position(price)
    assert result.iloc[:20].isna().all()
    assert not result.iloc[20:].isna().any()


# ===========================================================================
# 4. TRB signal
# ===========================================================================


def test_trb_signal_breakout_up():
    """
    With price[0..21]=10, price[22]=15. At t=23:
      P_{t-1} = P_22 = 15
      window for max/min: P_{t-21..t-2} = P_2..P_21 = [10]*20
      P_{t-1} = 15 > max = 10 ⇒ signal = +1
    """
    prices = [10.0] * 22 + [15.0, 10.0]  # P_23 = 10 won't affect TRB[23]
    price = pd.Series(prices, index=_bdates(24))
    result = compute_trb_signal(price)
    assert result.iloc[23] == 1.0


def test_trb_signal_breakdown():
    """
    With price[0..23]=10, price[24]=5. At t=25:
      P_{t-1} = P_24 = 5
      window for max/min: P_4..P_23 = [10]*20
      P_{t-1} = 5 < min = 10 ⇒ signal = -1
    """
    prices = [10.0] * 24 + [5.0, 7.0]
    price = pd.Series(prices, index=_bdates(26))
    result = compute_trb_signal(price)
    assert result.iloc[25] == -1.0


def test_trb_signal_in_range():
    """
    All constant ⇒ P_{t-1} == max == min ⇒ signal = 0 (neither
    strict > nor strict <).
    """
    price = pd.Series([10.0] * 30, index=_bdates(30))
    result = compute_trb_signal(price)
    assert result.iloc[21:].eq(0.0).all()


def test_trb_signal_warmup_pattern():
    """First 21 rows NaN: need P_{t-21..t-2} = 20 values via shift(2)+rolling(20)."""
    price = pd.Series(np.linspace(10.0, 20.0, 30), index=_bdates(30))
    result = compute_trb_signal(price)
    assert result.iloc[:21].isna().all()
    assert not result.iloc[21:].isna().any()


# ===========================================================================
# 5. RSI(14) Wilder
# ===========================================================================


def test_rsi_wilder_alternating_pattern_golden():
    """
    Worked example, Wilder (1978) algorithm. With 17 alternating prices
    [10, 11, 10, 11, ..., 11, 10]:

      deltas (positions 1..16):  +1, -1, +1, -1, ..., -1
      gains  (positions 1..16):   1,  0,  1,  0, ...,  0
      losses (positions 1..16):   0,  1,  0,  1, ...,  1

    At position 14 (initial simple mean of positions 1..14):
      AvgG_14 = (7 gains of 1 + 7 of 0) / 14 = 7/14 = 0.5
      AvgL_14 = 7/14 = 0.5
      RS = 1, RSI = 100 - 100/2 = 50.0

    At position 15 (recursive):
      gain[15] = 1, loss[15] = 0
      AvgG_15 = (13 * 0.5 + 1)/14 = 7.5/14 = 15/28
      AvgL_15 = (13 * 0.5 + 0)/14 = 6.5/14 = 13/28
      RS = (15/28)/(13/28) = 15/13
      RSI = 100 - 100/(1 + 15/13) = 100 - 100*13/28 = 100 - 1300/28
          = (2800 - 1300)/28 = 1500/28 ≈ 53.5714286

    After anti-leak shift(1):
      feature[15] = RSI from position 14 = 50.0
      feature[16] = RSI from position 15 = 1500/28
    """
    prices = [10.0, 11.0] * 8 + [10.0]  # 17 values, alternating ending at 10
    price = pd.Series(prices, index=_bdates(17))
    result = compute_rsi_wilder(price, period=14)

    assert result.iloc[15] == pytest.approx(50.0, abs=1e-9)
    assert result.iloc[16] == pytest.approx(1500.0 / 28.0, rel=1e-9)


def test_rsi_wilder_constant_price_yields_nan():
    """Constant prices ⇒ all gains/losses = 0 ⇒ 0/0 ⇒ RSI = NaN."""
    price = pd.Series([10.0] * 30, index=_bdates(30))
    result = compute_rsi_wilder(price, period=14)
    assert result.iloc[15:].isna().all()


def test_rsi_wilder_monotonic_up_yields_100():
    """Strictly increasing prices ⇒ all losses=0 ⇒ RSI = 100."""
    price = pd.Series(np.arange(10.0, 40.0, 1.0), index=_bdates(30))
    result = compute_rsi_wilder(price, period=14)
    # After warmup (shift(1) means valid from index 15), all 100
    assert result.iloc[15:].eq(100.0).all()


def test_rsi_wilder_monotonic_down_yields_0():
    """Strictly decreasing prices ⇒ all gains=0 ⇒ RSI = 0."""
    price = pd.Series(np.arange(40.0, 10.0, -1.0), index=_bdates(30))
    result = compute_rsi_wilder(price, period=14)
    assert result.iloc[15:].eq(0.0).all()


def test_rsi_wilder_warmup_pattern():
    """First 15 rows NaN (period=14 + 1 anti-leak shift)."""
    rng = np.random.default_rng(42)
    price = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))), index=_bdates(50)
    )
    result = compute_rsi_wilder(price, period=14)
    assert result.iloc[:15].isna().all()
    assert not result.iloc[15:].isna().any()


def test_rsi_wilder_input_shorter_than_period_returns_all_nan():
    """If input length ≤ period, no valid output possible."""
    price = pd.Series([10.0, 11.0, 10.0, 11.0, 10.0], index=_bdates(5))
    result = compute_rsi_wilder(price, period=14)
    assert result.isna().all()


def test_rsi_wilder_nan_in_initial_window_returns_all_nan():
    """
    If a NaN falls within positions 1..period (the initial-mean window),
    Wilder's baseline cannot be established ⇒ output is all NaN. This
    exercises the defensive early-return in `_wilder_smooth`.
    """
    prices = [10.0, 11.0, np.nan] + [10.0, 11.0] * 8  # NaN at index 2
    price = pd.Series(prices, index=_bdates(len(prices)))
    result = compute_rsi_wilder(price, period=14)
    assert result.isna().all()


# ===========================================================================
# 6. MACD normalized
# ===========================================================================


def test_macd_normalized_constant_price_yields_zero():
    """Constant prices ⇒ EMA_12 == EMA_26 == P ⇒ MACD = 0 (after warmup)."""
    price = pd.Series([10.0] * 35, index=_bdates(35))
    result = compute_macd_normalized(price)
    # min_periods=26 on slow EMA ⇒ EMAs valid from index 25, MACD raw valid
    # from 25, after shift(1) feature valid from index 26.
    assert result.iloc[:26].isna().all()
    assert result.iloc[26:].eq(0.0).all()


def test_macd_normalized_warmup_pattern():
    """First 26 rows NaN due to min_periods=26 on slow EMA + shift(1)."""
    price = pd.Series(np.linspace(10.0, 20.0, 40), index=_bdates(40))
    result = compute_macd_normalized(price)
    assert result.iloc[:26].isna().all()
    assert not result.iloc[26:].isna().any()


# ===========================================================================
# 7. compute_l2_features — top-level composer
# ===========================================================================


@pytest.fixture
def long_price_df_300day() -> pd.DataFrame:
    """300 business days, lognormal random walk (deterministic seed)."""
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    return pd.DataFrame({"adj_close": prices}, index=_bdates(300))


def test_compute_l2_features_returns_expected_columns(long_price_df_300day):
    result = compute_l2_features(long_price_df_300day)
    assert tuple(result.columns) == L2_FEATURE_COLS


def test_compute_l2_features_preserves_index(long_price_df_300day):
    result = compute_l2_features(long_price_df_300day)
    assert result.index.equals(long_price_df_300day.index)


def test_compute_l2_features_no_nan_after_full_warmup(long_price_df_300day):
    """After max warmup (momentum_12_3 needs 252 rows), no NaN expected."""
    result = compute_l2_features(long_price_df_300day)
    # First 252 rows: at least momentum is NaN
    assert result["momentum_12_3"].iloc[:252].isna().all()
    # After row 252: everything should be valid
    tail = result.iloc[252:]
    assert not tail.isna().any().any()


def test_compute_l2_features_custom_price_col():
    """`price_col` param honoured."""
    idx = _bdates(300)
    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = pd.DataFrame({"close_raw": prices}, index=idx)
    result = compute_l2_features(df, price_col="close_raw")
    assert tuple(result.columns) == L2_FEATURE_COLS


def test_compute_l2_features_raises_on_missing_col(long_price_df_300day):
    with pytest.raises(ValueError, match="price_col"):
        compute_l2_features(long_price_df_300day, price_col="nope")


def test_compute_l2_features_raises_on_non_positive():
    prices = [10.0] * 299 + [0.0]
    df = pd.DataFrame({"adj_close": prices}, index=_bdates(300))
    with pytest.raises(ValueError, match="non-positive"):
        compute_l2_features(df)


# ===========================================================================
# 8. Anti-leak (Tier 1) — global guarantee across all L2 indicators
# ===========================================================================


def test_anti_leak_mutating_last_price_does_not_change_last_row(long_price_df_300day):
    """
    CRITICAL Tier 1: every L2 feature at the last row must depend only
    on P_{t-1} and earlier. Mutating P_t (the last row's price) must
    NOT change any feature value at the last row.
    """
    original = compute_l2_features(long_price_df_300day)

    mutated_df = long_price_df_300day.copy()
    last_idx = mutated_df.index[-1]
    mutated_df.loc[last_idx, "adj_close"] = 9999.0  # massive perturbation

    mutated = compute_l2_features(mutated_df)

    pd.testing.assert_series_equal(
        original.iloc[-1], mutated.iloc[-1], check_names=False
    )


def test_anti_leak_future_nan_does_not_poison_past():
    """
    Stronger Tier 1: replace ALL prices at index ≥ 253 with NaN. The
    row at index 252 must match between the full input and the
    NaN-suffix input — it sees only P_0..P_251 in both cases.
    """
    n = 280
    rng = np.random.default_rng(42)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    full_df = pd.DataFrame({"adj_close": base}, index=_bdates(n))

    suffix_nan = base.copy()
    suffix_nan[253:] = np.nan
    nan_df = pd.DataFrame({"adj_close": suffix_nan}, index=_bdates(n))

    # validate() in compute_l2_features allows NaN (skips them via .dropna())
    full_result = compute_l2_features(full_df)
    nan_result = compute_l2_features(nan_df)

    pd.testing.assert_series_equal(
        full_result.iloc[252], nan_result.iloc[252], check_names=False
    )   