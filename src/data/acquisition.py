"""
Channel A: Daily APIs auto-fetch (vnstock + yfinance).

Phase 1 — static one-shot fetch:
- fetch_tcb_price: TCB OHLCV + adjusted close từ vnstock VCI
- fetch_vnindex: VN-Index OHLC từ vnstock
- fetch_usdvnd: USD/VND tỷ giá từ yfinance

Mọi function trả DataFrame conform với schema lock trong schema.py
và đã pass validation từ validation.py.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz

from .schema import (
    TCB_PRICE_SCHEMA,
    USDVND_SCHEMA,
    VNINDEX_SCHEMA,
)
from .validation import (
    canonicalize_price_unit,
    check_calendar_gaps,
    check_price_monotonicity,
)

logger = logging.getLogger(__name__)

ICT = pytz.timezone("Asia/Ho_Chi_Minh")
PROJECT_START = "2018-06-04"


def _now_ict() -> pd.Timestamp:
    """Current timestamp in Asia/Ho_Chi_Minh, suitable for fetched_at column."""
    return pd.Timestamp.now(tz=ICT)


def _today_str() -> str:
    """Today as YYYY-MM-DD string in ICT."""
    return datetime.now(ICT).strftime("%Y-%m-%d")


def _ensure_ohlcv_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize OHLCV column names + dtypes regardless of source."""
    # Drop yfinance-specific extras BEFORE rename. Lý do: yfinance với
    # auto_adjust=False trả cả 'Close' lẫn 'Adj Close' — nếu cả hai cùng
    # rename → 'close' sẽ tạo duplicate columns, khiến df['close'] trả về
    # DataFrame thay vì Series và pd.to_numeric raise TypeError.
    df = df.drop(
        columns=["Adj Close", "Dividends", "Stock Splits", "Capital Gains"],
        errors="ignore",
    )
    # Flatten MultiIndex columns nếu yfinance trả về ('Open', 'TICKER') tuples
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename_map = {
        "time": "date",
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(
            f"[{source_name}] Missing required OHLCV columns: {sorted(missing)}. "
            f"Got: {list(df.columns)}"
        )

    # Coerce dtypes. Handle cả tz-aware (yfinance) và tz-naive (vnstock).
    date_col = pd.to_datetime(df["date"])
    if getattr(date_col.dt, "tz", None) is not None:
        date_col = date_col.dt.tz_convert(None)
    df["date"] = date_col.dt.normalize()
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

    # Drop rows với close = NaN. Lý do: yfinance đôi khi trả về row hôm nay
    # với NaN (market chưa close) hoặc forex pair có gap NaN ngẫu nhiên.
    # Close là primary field — nếu NaN thì row không usable cho return calc.
    n_before = len(df)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(
            f"[{source_name}] Dropped {n_dropped} row(s) với close NaN "
            f"(thường là row hôm nay chưa close hoặc forex gap)"
        )

    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


# ============================================================
# TCB price (vnstock VCI)
# ============================================================

def fetch_tcb_price(
    start: str = PROJECT_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch TCB daily OHLCV + adjusted close từ vnstock VCI.

    Verified (Session 1 cũ): VCI trả về ADJUSTED close. 0/1985 ngày
    |log return|>15% mặc dù TCB stock dividend 1:1 năm 2024.
    Không cần custom adjustment logic.

    Output schema: tcb_price (date, open, high, low, close, volume, fetched_at)
    Unit: nghìn VND.
    """
    from vnstock import Vnstock

    end = end or _today_str()
    logger.info(f"[tcb_price] Fetching {start} → {end} via vnstock VCI")

    stock = Vnstock().stock(symbol="TCB", source="VCI")
    raw = stock.quote.history(start=start, end=end, interval="1D")

    df = _ensure_ohlcv_columns(raw, "tcb_price")
    df["fetched_at"] = _now_ict()
    df = df[[c.name for c in TCB_PRICE_SCHEMA.columns]]

    canonicalize_price_unit(df, expected_range=(5.0, 200.0))
    gap_report = check_calendar_gaps(df, "tcb_price")
    mono_report = check_price_monotonicity(df, "tcb_price")
    gap_report.log()
    mono_report.log()
    gap_report.raise_if_errors()

    TCB_PRICE_SCHEMA.validate(df)
    logger.info(f"[tcb_price] OK: {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ============================================================
# VN-Index (vnstock VCI)
# ============================================================

def fetch_vnindex(
    start: str = PROJECT_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch VN-Index daily OHLC từ vnstock.

    Output schema: vnindex (date, open, high, low, close, volume, fetched_at)
    Close là index points (không phải VND).
    """
    from vnstock import Vnstock

    end = end or _today_str()
    logger.info(f"[vnindex] Fetching {start} → {end} via vnstock")

    stock = Vnstock().stock(symbol="VNINDEX", source="VCI")
    raw = stock.quote.history(start=start, end=end, interval="1D")

    df = _ensure_ohlcv_columns(raw, "vnindex")
    df["fetched_at"] = _now_ict()
    df = df[[c.name for c in VNINDEX_SCHEMA.columns]]

    # VN-Index không apply canonicalize_price_unit (range khác — index points 800-1500)
    gap_report = check_calendar_gaps(df, "vnindex")
    # Monotonicity: VN-Index có thể có ngày biến động lớn (limit ~7%), giữ threshold 15%
    mono_report = check_price_monotonicity(df, "vnindex")
    gap_report.log()
    mono_report.log()
    gap_report.raise_if_errors()

    VNINDEX_SCHEMA.validate(df)
    logger.info(f"[vnindex] OK: {len(df)} rows")
    return df


# ============================================================
# USD/VND (yfinance)
# ============================================================

def fetch_usdvnd(
    start: str = PROJECT_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch USD/VND tỷ giá từ yfinance USDVND=X.

    Output schema: usdvnd (date, open, high, low, close, volume, fetched_at)
    Close là tỷ giá (~23000-26000). Open/High/Low có thể nullable nếu
    yfinance không cung cấp đầy đủ cho cặp FX.
    """
    import yfinance as yf

    end = end or _today_str()
    logger.info(f"[usdvnd] Fetching {start} → {end} via yfinance")

    ticker = yf.Ticker("USDVND=X")
    raw = ticker.history(start=start, end=end, interval="1d", auto_adjust=False)
    if raw.empty:
        raise RuntimeError("[usdvnd] yfinance returned empty DataFrame")

    raw = raw.reset_index()
    df = _ensure_ohlcv_columns(raw, "usdvnd")
    df["fetched_at"] = _now_ict()
    df = df[[c.name for c in USDVND_SCHEMA.columns]]

    # FX không có volume meaningful → set 0
    df["volume"] = 0

    # Sanity: USD/VND should be ~22000-28000 in 2018-2026
    median_close = float(df["close"].median())
    if not (15000 <= median_close <= 35000):
        raise ValueError(
            f"[usdvnd] Median close {median_close:.0f} ngoài expected range "
            f"[15000, 35000]. Source có thể đang trả unit khác."
        )

    gap_report = check_calendar_gaps(df, "usdvnd")
    gap_report.log()
    gap_report.raise_if_errors()

    USDVND_SCHEMA.validate(df)
    logger.info(f"[usdvnd] OK: {len(df)} rows")
    return df


# ============================================================
# Persistence
# ============================================================

def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet với pyarrow engine."""
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info(f"Saved {len(df)} rows → {path}")