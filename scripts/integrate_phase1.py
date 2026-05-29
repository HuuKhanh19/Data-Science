"""
scripts/integrate_phase1.py — runner Bước 4 (tích hợp dữ liệu).

Đọc 3 file daily ở `data/interim/` (đã trên spine, Bước 3) + 3 nguồn chậm ở
`data/raw/` -> tích hợp -> ghi `data/interim/integrated.parquet` + `_integrate_log.json`.
Trả exit code 0 nếu OK, 1 nếu lỗi (tiện tự động hóa Phase 2).

    python scripts/integrate_phase1.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import integrate as I  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw"
OUT = INTERIM / "integrated.parquet"
LOG = INTERIM / "_integrate_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _slow_value_cols(panel: pd.DataFrame) -> list[str]:
    """Cột giá trị của 3 nguồn chậm (loại daily + reference_period/release_date)."""
    daily = set(I._daily_cols()) | {"date", "fx_ffilled"}
    return [c for c in panel.columns
            if not (c in daily
                    or c.endswith("_reference_period")
                    or c.endswith("_release_date"))]


def main() -> int:
    try:
        spine   = pd.read_parquet(INTERIM / "tcb_price_clean.parquet")
        vnindex = pd.read_parquet(INTERIM / "vnindex_clean.parquet")
        usdvnd  = pd.read_parquet(INTERIM / "usdvnd_clean.parquet")
        cpi  = pd.read_parquet(RAW / "cpi.parquet")
        gdp  = pd.read_parquet(RAW / "gdp.parquet")
        fund = pd.read_parquet(RAW / "tcb_fundamentals.parquet")

        panel = I.integrate(spine, vnindex, usdvnd, cpi, gdp, fund)
        I.validate(panel, spine)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(OUT, index=False)

        # leading-NaN mỗi cột slow -> verify ranh giới warmup (npl/nim có prefix ~2018-Q1)
        leading_nan = {c: int(panel[c].isna().sum()) for c in _slow_value_cols(panel)}

        log = {
            "status": "ok",
            "output": str(OUT),
            "rows": len(panel),
            "n_cols": len(panel.columns),
            "cols": list(panel.columns),
            "date_min": str(panel["date"].min().date()),
            "date_max": str(panel["date"].max().date()),
            "leading_nan_slow": leading_nan,
            "checks": ["spine_aligned", "daily_complete",
                       "asof_no_leakage", "leading_nan_only"],
            "generated_at": datetime.now(TZ).isoformat(),
        }
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {len(panel)} phiên x {len(panel.columns)} cột -> {OUT}")
        print(f"     {log['date_min']} -> {log['date_max']}")
        print(f"     leading_nan (slow): {leading_nan}")
        return 0

    except Exception as e:  # noqa: BLE001
        err = {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "generated_at": datetime.now(TZ).isoformat(),
        }
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ERROR] {err['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())