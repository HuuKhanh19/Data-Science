"""Step 3 orchestrator: đọc data/raw/, làm sạch + căn spine HOSE, ghi data/interim/.

Output 3 file (theo quyết định "3 file riêng"):
    data/interim/tcb_price_clean.parquet   — full OHLCV, là spine
    data/interim/vnindex_clean.parquet     — full OHLCV, căn về spine
    data/interim/usdvnd_clean.parquet      — full OHLCV + cờ fx_ffilled, căn về spine + ffill

Cùng một _clean_log.json (audit + tiện cho tự động hoá Phase 2: exit code 0/1).

Usage:
    python scripts/clean_phase1.py
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.clean import align_fx, align_vnindex, build_spine, clean_tcb_price
from src.data.schema import TCB_PRICE_SCHEMA, VNINDEX_SCHEMA
from src.data.validation import check_monotonic_dates

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"


def _range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    return df["date"].min().date().isoformat(), df["date"].max().date().isoformat()


def main() -> int:
    print(f"Step 3 (clean + spine) starting at {datetime.now(TZ_VN).isoformat()}")
    print(f"  Raw dir    : {RAW}")
    print(f"  Interim dir: {INTERIM}")
    INTERIM.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    try:
        tcb_raw = pd.read_parquet(RAW / "tcb_price.parquet")
        vnindex_raw = pd.read_parquet(RAW / "vnindex.parquet")
        usdvnd_raw = pd.read_parquet(RAW / "usdvnd.parquet")

        # 1) TCB + spine -------------------------------------------------------
        tcb = clean_tcb_price(tcb_raw)
        spine = build_spine(tcb)
        TCB_PRICE_SCHEMA.validate(tcb)
        rep = check_monotonic_dates(tcb, "tcb_price_clean")
        if not rep.ok:
            raise ValueError(rep.summary())
        tcb.to_parquet(INTERIM / "tcb_price_clean.parquet", index=False)
        d0, d1 = _range(tcb)
        results["tcb_price_clean"] = {"status": "ok", "rows": len(tcb),
                                      "date_min": d0, "date_max": d1, "spine_len": len(spine)}
        print(f"  ✓ tcb_price_clean: {len(tcb)} phiên (spine) range=({d0} → {d1})")

        # 2) VNINDEX căn spine -------------------------------------------------
        vni = align_vnindex(vnindex_raw, spine)
        VNINDEX_SCHEMA.validate(vni)
        rep = check_monotonic_dates(vni, "vnindex_clean")
        if not rep.ok:
            raise ValueError(rep.summary())
        vni.to_parquet(INTERIM / "vnindex_clean.parquet", index=False)
        d0, d1 = _range(vni)
        results["vnindex_clean"] = {"status": "ok", "rows": len(vni),
                                    "date_min": d0, "date_max": d1}
        print(f"  ✓ vnindex_clean: {len(vni)} phiên range=({d0} → {d1})")

        # 3) FX căn spine + ffill ---------------------------------------------
        fx = align_fx(usdvnd_raw, spine)
        n_ffilled = int(fx["fx_ffilled"].sum())
        n_leading_nan = int(fx["close"].isna().sum())
        fx.to_parquet(INTERIM / "usdvnd_clean.parquet", index=False)
        d0, d1 = _range(fx)
        results["usdvnd_clean"] = {"status": "ok", "rows": len(fx),
                                   "date_min": d0, "date_max": d1,
                                   "n_ffilled": n_ffilled, "n_leading_nan": n_leading_nan}
        print(f"  ✓ usdvnd_clean: {len(fx)} phiên range=({d0} → {d1}) "
              f"| ffill={n_ffilled} | leading_nan={n_leading_nan}")

        if n_leading_nan > 0:
            print(f"  ⚠ FX có {n_leading_nan} phiên đầu NaN (trước quan sát FX đầu tiên) — "
                  f"để nguyên, sẽ cắt ở warmup Bước 7.")

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        traceback.print_exc()
        results["_error"] = {"status": "error", "error": str(e), "error_type": type(e).__name__}

    log = {"started_at": datetime.now(TZ_VN).isoformat(), "results": results}
    with open(INTERIM / "_clean_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n=== Log written to {INTERIM / '_clean_log.json'} ===")

    n_err = sum(1 for r in results.values() if r.get("status") == "error")
    print(f"=== Summary: {len(results)} outputs, {n_err} errors ===")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())