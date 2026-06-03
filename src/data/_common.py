"""Helper dùng chung cho các module fetch_* của Step 1.

Gom các tiện ích từng bị lặp ở fetch_prices/fetch_macro/fetch_fundamentals về một chỗ:
  - _now_vn:          timestamp giờ VN cho cột fetched_at
  - _http_get_bytes:  GET có retry + User-Agent (dùng cho scrape GDP)
  - _normalize_ohlcv: chuẩn hóa khung OHLCV (cả vnstock lẫn yfinance)
  - _save_and_report: validate schema + chạy validators → ghi parquet → dict báo cáo
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import urllib3

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
OHLCV_ORDER = ["date", "open", "high", "low", "close", "volume", "fetched_at"]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 30


def _now_vn() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(TZ_VN))


def _http_get_bytes(url: str, retries: int = 3, timeout: int = DEFAULT_TIMEOUT,
                    verify_ssl: bool = True) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            last_err = e
            print(f"  [attempt {attempt+1}/{retries}] GET {url} failed: {e}")
    raise RuntimeError(f"All {retries} attempts failed for {url}: {last_err}")


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
            raise ValueError(
                f"Required column '{c}' missing after normalize. "
                f"Have: {list(d.columns)}")

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