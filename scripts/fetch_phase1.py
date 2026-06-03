"""Phase 1 orchestrator: chạy 3 channels, save 6 parquets, ghi _fetch_log.json.

Usage:
    python scripts/fetch_phase1.py
"""
from __future__ import annotations
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.fetch_prices import fetch_tcb_price, fetch_vnindex
from src.data.fetch_fx import fetch_usdvnd
from src.data.fetch_cpi import fetch_cpi
from src.data.fetch_gdp import fetch_gdp
from src.data.fetch_fundamentals import fetch_tcb_fundamentals

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")

START_DATE = "2018-06-04"
WARMUP_START = "2017-01-01"          
END_DATE = datetime.now(TZ_VN).date().isoformat()

OUT_DIR = ROOT / "data" / "raw"


def _run_one(name: str, fn: Callable, **kwargs) -> Dict[str, Any]:
    print(f"\n=== [{name}] Fetching ===")
    try:
        result = fn(**kwargs)
        print(f"  → status={result.get('status')} rows={result.get('rows')} "
              f"range=({result.get('date_min')} → {result.get('date_max')})")
        return result
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }


def main() -> int:
    print(f"Phase 1 fetch starting at {datetime.now(TZ_VN).isoformat()}")
    print(f"  Range: {START_DATE} → {END_DATE}")
    print(f"  Output dir: {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # Channel A — daily prices
    results["tcb_price"] = _run_one(
        "tcb_price", fetch_tcb_price,
        start_date=START_DATE, end_date=END_DATE,
        out_path=OUT_DIR / "tcb_price.parquet",
    )
    results["vnindex"] = _run_one(
        "vnindex", fetch_vnindex,
        start_date=START_DATE, end_date=END_DATE,
        out_path=OUT_DIR / "vnindex.parquet",
    )
    results["usdvnd"] = _run_one(
        "usdvnd", fetch_usdvnd,
        start_date=START_DATE, end_date=END_DATE,
        out_path=OUT_DIR / "usdvnd.parquet",
    )

    # Channel B — macro scrapes
    results["cpi"] = _run_one(
        "cpi", fetch_cpi,
        out_path=OUT_DIR / "cpi.parquet",
        start_date=START_DATE, end_date=END_DATE,
    )
    results["gdp"] = _run_one(
        "gdp", fetch_gdp,
        out_path=OUT_DIR / "gdp.parquet",
        start_date=WARMUP_START, end_date=END_DATE,      
    )
    results["tcb_fundamentals"] = _run_one(
        "tcb_fundamentals", fetch_tcb_fundamentals,
        out_path=OUT_DIR / "tcb_fundamentals.parquet",
        start_date=WARMUP_START, end_date=END_DATE,       
    )

    log_path = OUT_DIR / "_fetch_log.json"
    log = {
        "started_at": datetime.now(TZ_VN).isoformat(),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "results": results,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n=== Log written to {log_path} ===")

    n_ok = sum(1 for r in results.values() if r.get("status") == "ok")
    n_warn = sum(1 for r in results.values() if r.get("status") == "warning")
    n_err = len(results) - n_ok - n_warn
    print(f"\n=== Summary: {n_ok} OK, {n_warn} warning, {n_err} errors ===")
    for name, r in results.items():
        if r.get("status") == "ok":
            flag = "✓"
        elif r.get("status") == "warning":
            flag = "⚠"
        else:
            flag = "✗"
        rows = r.get("rows", "—")
        print(f"  {flag} {name}: rows={rows}")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())