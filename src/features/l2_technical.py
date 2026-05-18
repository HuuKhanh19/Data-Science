"""
L2 — Technical indicators (research_design.md §4.2).

Six features computed daily from the adjusted close price:

    ma_crossover   = MA_5(P)_{t-1} / MA_20(P)_{t-1}
    momentum_12_3  = log(P_{t-63} / P_{t-252})            # J-T skip-3 (D7=B)
    bb_position    = (P_{t-1} - MA_20_{t-1}) / (2 * std_20_{t-1})
    trb_signal     = 1{P_{t-1} > max(P_{t-21..t-2})} - 1{P_{t-1} < min(P_{t-21..t-2})}
    rsi14          = Wilder's RSI(14), anti-leak shifted (decision D5)
    macd_norm      = (EMA_12(P) - EMA_26(P)) / P_{t-1},
                     EMA computed with pandas .ewm(adjust=False) (decision D6)

Anti-leakage convention (research_design.md §4.1, §2.5)
-------------------------------------------------------
Every indicator at trading day ``t`` uses information only from days
``t-1`` and earlier. This is achieved by either (a) shifting the input
price series by 1 before the rolling operation, or (b) computing the
indicator on the full series and shifting the output by 1. Both are
information-equivalent for the indicators here (all are
shift-equivariant: their value at time t depends only on prices at times
≤ t in the input, so shifting the input by k is equivalent to shifting
the output by k).

Bollinger Bands convention
--------------------------
Standard deviation uses ``ddof=0`` (population std), per Bollinger
(2001) and Fang/Jacobsen/Qin (2014) original definitions.

RSI(14) convention
------------------
Wilder's (1978) original exponential smoothing:
  - First valid output at index ``period`` (0-indexed) = simple mean of
    gains/losses at indices 1..period.
  - Subsequent: avg_n = ((period-1) * avg_{n-1} + value_n) / period.

This differs slightly from pandas' default ``ewm(alpha=1/14)``
initialization (which starts from the first valid value). The manual
implementation here matches Wilder's textbook exactly.

MACD convention
---------------
EMA fast=12, slow=26, no signal line (we use the raw difference
normalized by ``P_{t-1}``). ``adjust=False`` matches the canonical
recursive EMA used by TA-Lib, TradingView, and most trading platforms.
A minimum warmup of 26 periods is enforced via ``min_periods=26`` on
the slow EMA so that early under-converged values are NaN.

Warmup
------
- ma_crossover: 21 rows NaN (MA_20 + 1 shift)
- momentum_12_3: 252 rows NaN
- bb_position: 21 rows NaN
- trb_signal: 21 rows NaN (need P_{t-21..t-2} = 20 values shifted by 2)
- rsi14: 15 rows NaN (period=14 + 1 anti-leak shift)
- macd_norm: 27 rows NaN (slow EMA span=26 + 1 anti-leak shift)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# Public column names and frozen hyperparameters.
L2_FEATURE_COLS: tuple[str, ...] = (
    "ma_crossover",
    "momentum_12_3",
    "bb_position",
    "trb_signal",
    "rsi14",
    "macd_norm",
)

MA_SHORT: int = 5
MA_LONG: int = 20
BB_WINDOW: int = 20
TRB_WINDOW: int = 20  # max/min over P_{t-21..t-2} = 20 values
MOMENTUM_SHORT_SKIP: int = 63    # ~3 trading-month skip (decision D7=B)
MOMENTUM_LONG_LOOKBACK: int = 252  # ~12 trading months
RSI_PERIOD: int = 14
MACD_SPAN_FAST: int = 12
MACD_SPAN_SLOW: int = 26


# ---------------------------------------------------------------------------
# Public top-level composer
# ---------------------------------------------------------------------------


def compute_l2_features(
    price_df: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Compute all six L2 technical indicators."""
    if price_col not in price_df.columns:
        raise ValueError(
            f"price_col '{price_col}' not in price_df.columns: "
            f"{price_df.columns.tolist()}"
        )
    price = price_df[price_col]
    valid = price.dropna()
    if (valid <= 0).any():
        bad = valid[valid <= 0]
        raise ValueError(
            f"price_df['{price_col}'] contains {len(bad)} non-positive values"
        )

    result = pd.DataFrame(
        {
            "ma_crossover":  compute_ma_crossover(price),
            "momentum_12_3": compute_momentum_12_3(price),
            "bb_position":   compute_bb_position(price),
            "trb_signal":    compute_trb_signal(price),
            "rsi14":         compute_rsi_wilder(price, period=RSI_PERIOD),
            "macd_norm":     compute_macd_normalized(price),
        },
        index=price_df.index,
    )

    log.info(
        "compute_l2_features: %d rows. Warmup NaN by feature: %s",
        len(result),
        {c: int(result[c].isna().sum()) for c in result.columns},
    )
    return result


# ---------------------------------------------------------------------------
# Individual indicators (each is unit-testable)
# ---------------------------------------------------------------------------


def compute_ma_crossover(price: pd.Series) -> pd.Series:
    """
    MA crossover ratio: MA_5 / MA_20 using prices through P_{t-1}.

    Brock, Lakonishok & LeBaron (1992), Journal of Finance.
    """
    ma_short = price.rolling(MA_SHORT).mean().shift(1)
    ma_long = price.rolling(MA_LONG).mean().shift(1)
    return ma_short / ma_long


def compute_momentum_12_3(price: pd.Series) -> pd.Series:
    """
    Momentum 3-to-12 month: log(P_{t-63} / P_{t-252}).

    Decision D7 (Session 3) = B: skip-3 J-T convention. Captures pure
    medium-term momentum, excludes recent 3-month period to avoid
    short-term reversal contamination.

    Jegadeesh & Titman (1993), Journal of Finance.
    """
    log_p = np.log(price)
    return log_p.shift(MOMENTUM_SHORT_SKIP) - log_p.shift(MOMENTUM_LONG_LOOKBACK)


def compute_bb_position(price: pd.Series) -> pd.Series:
    """
    Bollinger Band position: (P_{t-1} - MA_20) / (2 * std_20).

    Uses population std (ddof=0) per Bollinger (2001) original
    convention. MA and std computed over the same 20-day window ending
    at P_{t-1}.

    Fang, Jacobsen & Qin (2014).
    """
    ma_20 = price.rolling(BB_WINDOW).mean().shift(1)
    std_20 = price.rolling(BB_WINDOW).std(ddof=0).shift(1)
    p_prev = price.shift(1)
    # If std_20 == 0 (constant window), BB position is undefined ⇒ NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        return (p_prev - ma_20) / (2 * std_20)


def compute_trb_signal(price: pd.Series) -> pd.Series:
    """
    Trading Range Breakout signal ∈ {-1, 0, 1, NaN}.

    +1 if P_{t-1} > max(P_{t-21..t-2})
    -1 if P_{t-1} < min(P_{t-21..t-2})
     0 otherwise.
    NaN during warmup (need P_{t-21} available).

    Brock, Lakonishok & LeBaron (1992).
    """
    p_prev = price.shift(1)
    # Rolling over price.shift(2): at index t, includes P_{t-21..t-2}.
    rolling_max = price.shift(2).rolling(TRB_WINDOW).max()
    rolling_min = price.shift(2).rolling(TRB_WINDOW).min()

    up = (p_prev > rolling_max).astype(float)
    down = (p_prev < rolling_min).astype(float)
    signal = up - down

    # Restore NaN where any input was NaN (astype(float) on a bool series
    # from a NaN comparison gives 0.0, not NaN — explicit mask below).
    invalid = p_prev.isna() | rolling_max.isna() | rolling_min.isna()
    signal[invalid] = np.nan
    return signal


def compute_rsi_wilder(price: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI with Wilder's exponential smoothing (1978), anti-leak shifted.

    Output at trading day t is the RSI value computed using price
    changes through P_{t-1}.

    Edge-case conventions:
      - avg_loss == 0, avg_gain > 0   ⇒ RS = +∞ ⇒ RSI = 100
      - avg_loss == 0, avg_gain == 0  ⇒ RS undefined ⇒ RSI = NaN
      - avg_gain == 0, avg_loss > 0   ⇒ RS = 0 ⇒ RSI = 0

    Citation: Panigrahi et al. (2021) per research_design.md §4.2;
    canonical algorithm from Wilder (1978) "New Concepts in Technical
    Trading Systems".
    """
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)

    # Anti-leak shift: at day t use the RSI value formed through P_{t-1}.
    return rsi.shift(1)


def compute_macd_normalized(
    price: pd.Series,
    span_fast: int = MACD_SPAN_FAST,
    span_slow: int = MACD_SPAN_SLOW,
) -> pd.Series:
    """
    Normalized MACD: (EMA_12(P) - EMA_26(P)) / P_{t-1}, anti-leak shifted.

    Normalization by price level removes the non-stationary scale
    dependence of raw MACD (Wang & Kim 2018).

    EMA uses ``adjust=False`` (canonical recursive form) and
    ``min_periods=span_slow`` to suppress under-converged early values.
    """
    ema_fast = price.ewm(span=span_fast, adjust=False, min_periods=span_slow).mean()
    ema_slow = price.ewm(span=span_slow, adjust=False, min_periods=span_slow).mean()
    macd_raw = (ema_fast - ema_slow) / price
    return macd_raw.shift(1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """
    Wilder (1978) exponential smoothing.

    Convention here matches the RSI use case: ``values`` has NaN at
    index 0 (it is typically a ``diff()`` output) and valid entries
    afterwards. The simple-mean initialization averages
    ``values.iloc[1:period+1]`` (``period`` valid entries) and places
    the result at index ``period``. Subsequent indices use the
    recursive formula.

    For non-RSI use cases (no leading NaN), this offsets the first valid
    output by one — that mismatch is documented and the function is
    private (used only from ``compute_rsi_wilder``).
    """
    n = len(values)
    out = np.full(n, np.nan)

    if n <= period:
        return pd.Series(out, index=values.index)

    initial_window = values.iloc[1 : period + 1]
    if initial_window.isna().any():
        # Cannot establish baseline ⇒ no valid output.
        return pd.Series(out, index=values.index)

    out[period] = initial_window.mean()

    arr = values.to_numpy(dtype=float)
    for i in range(period + 1, n):
        if np.isnan(arr[i]):
            out[i] = np.nan
        else:
            out[i] = (out[i - 1] * (period - 1) + arr[i]) / period

    return pd.Series(out, index=values.index)