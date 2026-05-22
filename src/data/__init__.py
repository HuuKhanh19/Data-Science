"""
src.data — Bước 1: Data Acquisition.

Public API for Phase 1 (static one-shot fetch):

Channel A (automated):
    fetch_tcb_price, fetch_vnindex, fetch_usdvnd

Channel B (CSV-based):
    load_macro_csv

Channel C (CSV-based, with optional vnstock cross-check):
    load_tcb_fundamentals_csv

Persistence:
    save_parquet
"""
from .acquisition import (
    PROJECT_START,
    fetch_tcb_price,
    fetch_usdvnd,
    fetch_vnindex,
    save_parquet,
)
from .manual_input import load_macro_csv, load_tcb_fundamentals_csv
from .schema import ALL_SCHEMAS
from .validation import (
    canonicalize_price_unit,
    check_calendar_gaps,
    check_price_monotonicity,
)

__all__ = [
    "PROJECT_START",
    "fetch_tcb_price",
    "fetch_vnindex",
    "fetch_usdvnd",
    "save_parquet",
    "load_macro_csv",
    "load_tcb_fundamentals_csv",
    "ALL_SCHEMAS",
    "canonicalize_price_unit",
    "check_calendar_gaps",
    "check_price_monotonicity",
]