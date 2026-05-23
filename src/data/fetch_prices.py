"""Channel A — Daily prices.

- fetch_tcb_price: vnstock Quote(source='VCI')
- fetch_vnindex: vnstock Quote(source='VCI')
- fetch_usdvnd: yfinance Ticker('USDVND=X') với dropna(close) fix

Tất cả output OHLCV schema chuẩn + fetched_at tz-aware Asia/Ho_Chi_Minh.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from .schema import TCB_PRICE_SCHEMA, USDVND_SCHEMA, VNINDEX_SCHEMA
from .validation import (
    canonicalize_price_unit, check_abnormal_returns,
    check_hose_calendar_gap, check_monotonic_dates,
)

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
OHLCV_ORDER = ["date", "open", "high", "low", "close", "volume", "fetched_at"]


def _now_vn() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(TZ_VN))


def _normalize_ohlcv(df: pd.DataFrame, rename_map: Dict[str, str] | None = None) -> pd.DataFrame:
    """Chuẩn hóa columns OHLCV. Drop tz-aware index nếu có."""
    d = df.copy()

    # If date là index, reset
    if isinstance(d.index, pd.DatetimeIndex):
        idx_name = d.index.name or "date"
        d.index.name = idx_name
        # Drop tz nếu tz-aware (yfinance)
        if d.index.tz is not None:
            d.index = d.index.tz_convert(None)
        d = d.reset_index().rename(columns={idx_name: "date"})

    if rename_map:
        d = d.rename(columns=rename_map)

    # yfinance trả cả Close + Adj Close → drop trước rename để tránh duplicate
    for unwanted in ("Adj Close", "Dividends", "Stock Splits", "Capital Gains"):
        if unwanted in d.columns:
            d = d.drop(columns=[unwanted])

    # Standardize column names lowercase
    d.columns = [c.lower() if isinstance(c, str) else c for c in d.columns]

    # Ensure required columns
    for c in ("date", "close"):
        if c not in d.columns:
            raise ValueError(f"Required column '{c}' missing after normalize. Have: {list(d.columns)}")

    for c in ("open", "high", "low", "volume"):
        if c not in d.columns:
            d[c] = pd.NA

    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    for c in ("open", "high", "low"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0).astype("int64")

    # Drop rows where close is null (yfinance sometimes returns NaN cho row "today")
    d = d.dropna(subset=["close"])

    d = d.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    d["fetched_at"] = _now_vn()
    return d[OHLCV_ORDER]


def _save_and_report(df: pd.DataFrame, name: str, schema, out_path: Path,
                     validators: list) -> Dict[str, Any]:
    schema.validate(df)
    reports = [v(df, name) for v in validators]
    warns = sum(len(r.warnings) for r in reports)
    errs = sum(len(r.errors) for r in reports)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "status": "ok" if errs == 0 else "error",
        "rows": int(len(df)),
        "date_min": df["date"].min().date().isoformat() if not df.empty else None,
        "date_max": df["date"].max().date().isoformat() if not df.empty else None,
        "warnings": warns,
        "errors": errs,
        "validation_summary": [r.summary() for r in reports],
        "output": str(out_path),
    }


def fetch_tcb_price(start_date: str, end_date: str, out_path: Path) -> Dict[str, Any]:
    """vnstock VCI returns adjusted close. Verified: 0/1985 ngày |log return|>15%."""
    from vnstock import Quote
    q = Quote(symbol="TCB", source="VCI")
    raw = q.history(start=start_date, end=end_date, interval="1D")

    df = _normalize_ohlcv(raw, rename_map={"time": "date"})

    # TCB nghìn VND: median ~15-50
    canonicalize_price_unit(df, expected_range=(5.0, 200.0))

    return _save_and_report(
        df, "tcb_price", TCB_PRICE_SCHEMA, out_path,
        validators=[
            check_monotonic_dates,
            check_hose_calendar_gap,
            check_abnormal_returns,
        ],
    )


def fetch_vnindex(start_date: str, end_date: str, out_path: Path) -> Dict[str, Any]:
    from vnstock import Quote
    q = Quote(symbol="VNINDEX", source="VCI")
    raw = q.history(start=start_date, end=end_date, interval="1D")

    df = _normalize_ohlcv(raw, rename_map={"time": "date"})

    # VN-Index typical range 800-1500 → bypass unit check
    return _save_and_report(
        df, "vnindex", VNINDEX_SCHEMA, out_path,
        validators=[
            check_monotonic_dates,
            check_hose_calendar_gap,
            # Skip abnormal_returns: index moves > 15% rất hiếm, không cần threshold strict
        ],
    )


def fetch_usdvnd(start_date: str, end_date: str, out_path: Path) -> Dict[str, Any]:
    """yfinance USDVND=X. Lưu ý: yfinance đôi khi trả NaN close cho row 'today'
    → đã dropna trong _normalize_ohlcv.
    """
    import yfinance as yf
    t = yf.Ticker("USDVND=X")
    raw = t.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)

    df = _normalize_ohlcv(raw)

    # USD/VND typical ~22,000-27,000 (raw VND/USD). Không canonicalize unit.
    return _save_and_report(
        df, "usdvnd", USDVND_SCHEMA, out_path,
        validators=[
            check_monotonic_dates,
            # USDVND có lịch khác HOSE (FX 24/5) → không check_hose_calendar_gap
        ],
    )