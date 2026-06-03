"""Thu thập giá daily từ vnstock (VCI) — adjusted close.

  - fetch_tcb_price: vnstock Quote(symbol='TCB',     source='VCI')
  - fetch_vnindex:   vnstock Quote(symbol='VNINDEX', source='VCI')

Cả hai trả adjusted close (đã xử lý corporate action), đẩy qua _normalize_ohlcv
→ OHLCV schema chuẩn. USD/VND tách sang fetch_fx.py vì dùng yfinance, không phải
vnstock — một file một cơ chế.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from ._common import _normalize_ohlcv, _save_and_report
from .schema import TCB_PRICE_SCHEMA, VNINDEX_SCHEMA
from .validation import (
    canonicalize_price_unit, check_abnormal_returns,
    check_hose_calendar_gap, check_monotonic_dates,
)


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