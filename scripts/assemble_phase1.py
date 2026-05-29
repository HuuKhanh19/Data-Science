"""scripts/assemble_phase1.py — runner Step 7 (lắp ráp, xử lý NA, xuất artifact).

Đọc `data/interim/features_raw.parquet` (Step 5) + `data/interim/labels.parquet`
(Step 6) -> assemble (merge theo date + cắt warmup tự suy) -> validate (4-check) ->
ghi `data/processed/features.parquet` + `data/processed/_assemble_log.json`.
Exit 0 nếu OK, 1 nếu lỗi.

    python scripts/assemble_phase1.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import assemble as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
SRC_FEAT = INTERIM / "features_raw.parquet"
SRC_LABEL = INTERIM / "labels.parquet"
OUT = PROCESSED / "features.parquet"
LOG = PROCESSED / "_assemble_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def main() -> int:
    try:
        features_raw = pd.read_parquet(SRC_FEAT)
        labels = pd.read_parquet(SRC_LABEL)

        out = A.assemble(features_raw, labels)
        A.validate(out)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)

        n_raw = len(features_raw)
        n_final = len(out)
        tail_nan = {f"y_{k}": int(out[f"y_{k}"].isna().sum()) for k in A.HORIZONS}
        pct_pos = {
            f"y_{k}": round(
                100 * float((out[f"y_{k}"] == 1.0).sum())
                / int(out[f"y_{k}"].notna().sum()), 2
            )
            for k in A.HORIZONS
        }

        log = {
            "status": "ok",
            "output": str(OUT),
            "rows_raw": n_raw,
            "rows_final": n_final,
            "rows_warmup_cut": n_raw - n_final,
            "usable_start": str(out["date"].min().date()),
            "date_min": str(out["date"].min().date()),
            "date_max": str(out["date"].max().date()),
            "n_features": len(A.FEATURES),
            "n_labels": len(A.LABELS),
            "features": A.FEATURES,
            "labels": A.LABELS,
            "feature_nan_total": int(out[A.FEATURES].isna().to_numpy().sum()),
            "tail_nan": tail_nan,
            "pct_pos_usable": pct_pos,
            "checks": ["schema", "features_no_nan", "label_tail_nan", "date_key"],
            "generated_at": datetime.now(TZ).isoformat(),
        }
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {n_final} phiên (cắt {n_raw - n_final} warmup) "
              f"x {len(A.FEATURES)} feature + {len(A.LABELS)} nhãn -> {OUT}")
        print(f"     vùng khả dụng: {log['usable_start']} -> {log['date_max']}")
        print(f"     feature_nan_total={log['feature_nan_total']} | tail_nan={tail_nan}")
        print(f"     pct_pos(+1) vùng khả dụng: {pct_pos}")
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