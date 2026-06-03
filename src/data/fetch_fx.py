"""Thu thập tỷ giá USD/VND từ yfinance (USDVND=X).

Tách riêng khỏi fetch_prices vì cơ chế khác hẳn: nguồn là yfinance API
(không phải vnstock), và FX chạy lịch 24/5 (không phải lịch HOSE) → bỏ
check_hose_calendar_gap để không báo lỗi gap giả.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from ._common import _normalize_ohlcv, _save_and_report
from .schema import USDVND_SCHEMA
from .validation import check_monotonic_dates


def fetch_usdvnd(start_date: str, end_date: str, out_path: Path) -> Dict[str, Any]:
    """yfinance USDVND=X.

    Lưu ý: yfinance đôi khi trả NaN close cho row 'today'
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