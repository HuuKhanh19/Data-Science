"""
scripts/features_phase1.py — runner Bước 5 (feature engineering).

Đọc `data/interim/integrated.parquet` (Bước 4) + 3 nguồn chậm raw `data/raw/`
(cho YoY dùng history 2017) -> build 20 feature -> validate -> ghi
`data/interim/features_raw.parquet` + `_features_log.json`.
Trả exit code 0 nếu OK, 1 nếu lỗi (tiện tự động hóa Phase 2).

    python scripts/features_phase1.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import features as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw"
SRC = INTERIM / "integrated.parquet"
OUT = INTERIM / "features_raw.parquet"
LOG = INTERIM / "_features_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def main() -> int:
    try:
        panel = pd.read_parquet(SRC)
        cpi  = pd.read_parquet(RAW / "cpi.parquet")
        gdp  = pd.read_parquet(RAW / "gdp.parquet")
        fund = pd.read_parquet(RAW / "tcb_fundamentals.parquet")

        features = F.build_features(panel, cpi, gdp, fund)
        F.validate(features, panel, cpi, gdp, fund)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(OUT, index=False)

        # leading-NaN mỗi feature + mốc bắt đầu vùng khả dụng (Bước 7 sẽ cắt tới đây)
        leading_nan = {c: int(features[c].isna().sum()) for c in F.FEATURES}
        full = features[F.FEATURES].notna().all(axis=1)
        usable_start = str(features.loc[full.idxmax(), "date"].date()) if full.any() else None

        log = {
            "status": "ok",
            "output": str(OUT),
            "rows": len(features),
            "n_features": len(F.FEATURES),
            "features": F.FEATURES,
            "date_min": str(features["date"].min().date()),
            "date_max": str(features["date"].max().date()),
            "usable_start": usable_start,
            "leading_nan": leading_nan,
            "checks": ["spine_aligned", "feature_set",
                       "no_lookahead", "leading_nan_only"],
            "generated_at": datetime.now(TZ).isoformat(),
        }
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {len(features)} phiên x {len(F.FEATURES)} feature -> {OUT}")
        print(f"     {log['date_min']} -> {log['date_max']} | vùng khả dụng từ: {usable_start}")
        print(f"     leading_nan: {leading_nan}")
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