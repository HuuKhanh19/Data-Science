"""
Data fetchers for TCB project.

Three public functions, each fetches one data product:
  - fetch_tcb_price: TCB OHLCV from vnstock (primary) or yfinance (fallback)
  - fetch_vnindex:   VN-Index OHLC from vnstock (primary) or yfinance (fallback)
  - fetch_usdvnd:    USD/VND rate from yfinance (vnstock không có FX)

Design philosophy (per Session 1 discussion)
--------------------------------------------
This module performs ONLY structural data-quality checks — those derivable
from data definitions, not from empirical observation:

  - Mức giá trị đơn (Slide_Data_Science.md): missing values, domain violations
    (prices > 0, volume >= 0), dtype correctness.
  - Mức bản ghi: OHLC integrity (low ≤ open/close ≤ high).
  - Mức tập giá trị: DatetimeIndex uniqueness, monotonic sort.

THRESHOLD-based checks (cross-source agreement, trading-day gap) are
deferred to notebook ``notebooks/00_data_source_verification.ipynb`` where
thresholds are derived from observed data distribution. This avoids the
"naive decree" anti-pattern: setting tolerance numbers without empirical basis.

Per research_design.md section 2.5: "Không forward-fill giá hoặc returns".
Therefore missing OHLCV values in raw fetched data are an ERROR
(handled by validator), not silently imputed.

Open Q1 (IMPLEMENTATION.md section 11)
--------------------------------------
vnstock VCI returns a single ``close`` column without an explicit ``adj_close``.
Whether VCI's close is corporate-action-adjusted (split + dividend) is the
subject of Open Q1, resolved in notebook 00 by visual inspection at known
high-magnitude events. Until resolved, ``close`` and ``adj_close`` are populated
identically from VCI's output (consistent with research_design.md section 2.3
which lists vnstock VCI as the primary source for adjusted close).

For yfinance fallback, ``auto_adjust=True`` is used so that ALL of OHLC are
adjusted consistently. This is critical: mixing raw OHL with adjusted close
would violate ``low ≤ close ≤ high`` on every date preceding a corporate
action (because adj_close = raw_close × factor, with factor ≤ 1, so
adj_close < raw_low whenever a stock dividend/split occurred later).
Yfinance's ``auto_adjust=True`` returns Open/High/Low/Close all adjusted,
keeping integrity intact.

For full Open-Q1 forensic analysis (comparing vnstock vs yfinance RAW vs
yfinance ADJUSTED), notebook 00 calls yfinance directly with
``auto_adjust=False`` to expose both raw and adjusted series, bypassing this
wrapper.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# =====================================================================
# Internal: source-specific raw fetchers
# =====================================================================


def _fetch_ohlcv_vnstock(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV via vnstock VCI source.

    Returns DataFrame with DatetimeIndex (name='date') and columns:
        open, high, low, close, adj_close, volume.

    vnstock VCI returns: time, open, high, low, close, volume (no adj_close).
    We assume close = adj_close (verified in notebook 00).
    """
    log.info("vnstock VCI: fetching %s [%s → %s]", symbol, start, end)

    # Local import: avoid loading vnstock unless this function called
    # (vnstock import is heavy and has chart dependencies)
    from vnstock import Vnstock

    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        df = stock.quote.history(start=start, end=end, interval="1D", to_df=True)
    except Exception as e:
        raise RuntimeError(f"vnstock fetch failed for {symbol}: {e}") from e

    if df is None or len(df) == 0:
        raise RuntimeError(f"vnstock returned empty DataFrame for {symbol}")

    # Normalize column names and index
    if "time" not in df.columns:
        raise RuntimeError(
            f"vnstock returned unexpected columns for {symbol}: {df.columns.tolist()}"
        )

    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Strip timezone if present (we work with naive trading-day dates)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Populate adj_close = close (per Open Q1 assumption; verified in notebook 00)
    df["adj_close"] = df["close"]

    return df[["open", "high", "low", "close", "adj_close", "volume"]]


def _fetch_ohlcv_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV via yfinance.

    Returns DataFrame with DatetimeIndex (name='date') and columns:
        open, high, low, close, adj_close, volume.

    Uses ``auto_adjust=True`` so that ALL of Open/High/Low/Close are split- and
    dividend-adjusted consistently. This is required because returns are
    computed from adjusted prices (research_design.md 2.5) and OHLC integrity
    constraints (low ≤ close ≤ high etc.) only hold within a single
    adjustment basis — mixing raw OHL with adjusted close would generate
    spurious violations on dates preceding any corporate action.

    With auto_adjust=True, yfinance returns: Open, High, Low, Close, Volume
    (no separate "Adj Close" column, because Close IS the adjusted close).
    We populate ``adj_close = close`` for schema consistency with the vnstock
    branch and with the IMPLEMENTATION 4.1 schema.
    """
    log.info("yfinance: fetching %s [%s → %s]", ticker, start, end)

    import yfinance as yf

    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, interval="1d", auto_adjust=True)
    except Exception as e:
        raise RuntimeError(f"yfinance fetch failed for {ticker}: {e}") from e

    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned empty DataFrame for {ticker}")

    expected_yf_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected_yf_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"yfinance returned unexpected columns for {ticker}: missing {missing}, "
            f"got {df.columns.tolist()}"
        )

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df.index.name = "date"
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    # With auto_adjust=True, Close column IS the adjusted close.
    # Populate adj_close for schema consistency.
    df["adj_close"] = df["close"]

    return df[["open", "high", "low", "close", "adj_close", "volume"]]


# =====================================================================
# Internal: structural validators
# =====================================================================


def _validate_index(df: pd.DataFrame, name: str) -> None:
    """Check DatetimeIndex uniqueness and ascending sort."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: index is not DatetimeIndex (got {type(df.index).__name__})")
    if not df.index.is_unique:
        dups = df.index[df.index.duplicated()].tolist()
        raise ValueError(f"{name}: duplicate dates {dups[:5]}{'…' if len(dups) > 5 else ''}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}: dates not sorted ascending")


def _validate_no_nan(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Per research_design 2.5: NaN in prices/volume is an error, not silent fill."""
    nan_counts = df[cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts) > 0:
        raise ValueError(
            f"{name}: NaN values in raw fetched data (forward-fill is FORBIDDEN per "
            f"research_design 2.5): {nan_counts.to_dict()}"
        )


def _validate_ohlcv(df: pd.DataFrame, name: str) -> None:
    """Full structural validation for stock OHLCV (TCB)."""
    expected = {"open", "high", "low", "close", "adj_close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

    _validate_index(df, name)
    _validate_no_nan(df, list(expected), name)

    # Domain: prices strictly positive
    price_cols = ["open", "high", "low", "close", "adj_close"]
    bad_price = (df[price_cols] <= 0).any(axis=1)
    if bad_price.any():
        rows = df.index[bad_price].tolist()
        raise ValueError(f"{name}: non-positive prices at {rows[:5]}")

    # Domain: volume non-negative (zero-volume days possible on illiquid sessions)
    if (df["volume"] < 0).any():
        raise ValueError(f"{name}: negative volume detected")

    # Integrity: low ≤ {open, close, high}; high ≥ {open, close, low}
    bad_ohlc = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
    )
    if bad_ohlc.any():
        rows = df.index[bad_ohlc].tolist()
        raise ValueError(f"{name}: OHLC integrity violation at {rows[:5]}")


def _validate_index_price(df: pd.DataFrame, name: str) -> None:
    """Structural validation for market index (no volume)."""
    expected = {"close", "adj_close"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

    _validate_index(df, name)
    _validate_no_nan(df, list(expected), name)

    if (df[["close", "adj_close"]] <= 0).any().any():
        raise ValueError(f"{name}: non-positive index value")


def _validate_fx(df: pd.DataFrame, name: str) -> None:
    """Structural validation for FX rate (single column)."""
    if "rate" not in df.columns:
        raise ValueError(f"{name}: missing 'rate' column (got {df.columns.tolist()})")

    _validate_index(df, name)
    _validate_no_nan(df, ["rate"], name)

    if (df["rate"] <= 0).any():
        raise ValueError(f"{name}: non-positive FX rate")


# =====================================================================
# Public: top-level fetchers with primary→fallback logic
# =====================================================================

_SourceLiteral = Literal["auto", "vnstock", "yfinance"]


def _today_iso() -> str:
    return pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


def fetch_tcb_price(
    start: str = "2018-06-04",
    end: str | None = None,
    source: _SourceLiteral = "auto",
) -> pd.DataFrame:
    """
    Fetch TCB price OHLCV from vnstock (primary) or yfinance (fallback).

    Canonical output unit: **nghìn VND** (thousands of dong), matching the
    Vietnamese broker convention used by vnstock VCI. Yfinance returns prices
    in VND, so we divide by 1000 to standardize. This ensures the parquet
    file has consistent units regardless of which source succeeded — critical
    for cron production where source choice may vary day-to-day.

    Parameters
    ----------
    start : str
        ISO date "YYYY-MM-DD". Default: TCB listing date.
    end : str | None
        ISO date. None → today.
    source : {'auto', 'vnstock', 'yfinance'}
        'auto' (default) → try vnstock first, fall back to yfinance on error.
        Explicit choice → use only that source, raise on failure.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (name='date'), unique, sorted.
        Columns: open, high, low, close, adj_close, volume.
        Prices in nghìn VND (canonical project unit).
        All structural DQ checks passed.
    """
    end = end or _today_iso()

    if source in ("vnstock", "auto"):
        try:
            df = _fetch_ohlcv_vnstock(symbol="TCB", start=start, end=end)
            # vnstock VCI already in nghìn VND — no rescale needed
            _validate_ohlcv(df, name="TCB[vnstock]")
            log.info("TCB: vnstock returned %d rows (unit: nghìn VND)", len(df))
            return df
        except Exception as e:
            if source == "vnstock":
                raise
            log.warning("TCB: vnstock failed (%s); falling back to yfinance", e)

    df = _fetch_ohlcv_yfinance(ticker="TCB.VN", start=start, end=end)
    # yfinance returns VND; normalize to nghìn VND (canonical unit)
    _price_cols = ["open", "high", "low", "close", "adj_close"]
    df[_price_cols] = df[_price_cols] / 1000.0
    log.info("TCB: yfinance returned %d rows, scaled VND→nghìn VND", len(df))
    _validate_ohlcv(df, name="TCB[yfinance]")
    return df


def fetch_vnindex(
    start: str = "2018-06-04",
    end: str | None = None,
    source: _SourceLiteral = "auto",
) -> pd.DataFrame:
    """
    Fetch VN-Index from vnstock (primary, symbol='VNINDEX') or yfinance
    (fallback, ticker='^VNINDEX').

    Returns DataFrame with DatetimeIndex and columns: close, adj_close.
    Note: market indices don't have corporate-action adjustments at this level,
    so close == adj_close by definition.
    """
    end = end or _today_iso()

    if source in ("vnstock", "auto"):
        try:
            df = _fetch_ohlcv_vnstock(symbol="VNINDEX", start=start, end=end)
            df = df[["close", "adj_close"]]  # Drop O/H/L/volume for index schema
            _validate_index_price(df, name="VNINDEX[vnstock]")
            log.info("VNINDEX: vnstock returned %d rows", len(df))
            return df
        except Exception as e:
            if source == "vnstock":
                raise
            log.warning("VNINDEX: vnstock failed (%s); falling back to yfinance", e)

    df = _fetch_ohlcv_yfinance(ticker="^VNINDEX", start=start, end=end)
    df = df[["close", "adj_close"]]
    _validate_index_price(df, name="VNINDEX[yfinance]")
    log.info("VNINDEX: yfinance returned %d rows", len(df))
    return df


def fetch_usdvnd(
    start: str = "2018-06-04",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch USD/VND daily exchange rate from yfinance (ticker='USDVND=X').

    vnstock does not provide FX data, so no fallback.

    Returns DataFrame with DatetimeIndex and single column 'rate'.
    """
    end = end or _today_iso()

    df = _fetch_ohlcv_yfinance(ticker="USDVND=X", start=start, end=end)
    df = df[["adj_close"]].rename(columns={"adj_close": "rate"})
    _validate_fx(df, name="USDVND[yfinance]")
    log.info("USDVND: yfinance returned %d rows", len(df))
    return df


# =====================================================================
# Summary helpers (for acquire_data.py reporting, not validation)
# =====================================================================


def summary_stats(df: pd.DataFrame, name: str) -> dict:
    """
    Compute summary statistics for visual inspection.

    NOT a validator — does not raise. Purpose: surface anomalies to the human
    operator for manual inspection. Threshold-based decisions belong to
    notebook 00 (EDA), not here.

    Returns dict with keys:
        n_rows, date_min, date_max, max_calendar_gap_days,
        close_min, close_median, close_max, [volume_median if applicable]
    """
    stats: dict[str, object] = {
        "name": name,
        "n_rows": len(df),
        "date_min": str(df.index.min().date()),
        "date_max": str(df.index.max().date()),
    }

    # Max calendar-day gap between consecutive trading days.
    # Useful for spotting unusual data outages (e.g., > Tết duration).
    if len(df) >= 2:
        gaps = df.index.to_series().diff().dt.days.dropna()
        stats["max_calendar_gap_days"] = int(gaps.max())
        # gaps.idxmax() returns the index LABEL (a Timestamp) directly because
        # gaps is a Series with DatetimeIndex. Do NOT wrap in df.index[...] —
        # DatetimeIndex.__getitem__ accepts integer position, not Timestamp.
        stats["max_calendar_gap_at"] = str(gaps.idxmax().date())

    if "close" in df.columns:
        stats["close_min"] = float(df["close"].min())
        stats["close_median"] = float(df["close"].median())
        stats["close_max"] = float(df["close"].max())

    if "volume" in df.columns:
        stats["volume_median"] = float(df["volume"].median())
        stats["volume_zero_days"] = int((df["volume"] == 0).sum())

    if "rate" in df.columns:
        stats["rate_min"] = float(df["rate"].min())
        stats["rate_median"] = float(df["rate"].median())
        stats["rate_max"] = float(df["rate"].max())

    return stats