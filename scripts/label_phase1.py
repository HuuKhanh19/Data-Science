"""scripts/label_phase1.py — runner Bước 6 (gán nhãn).

Đọc `data/interim/integrated.parquet` (Bước 4; cột `close` = adjusted close trên
spine HOSE, non-null toàn spine) -> build 4 nhãn y_{t,k} -> validate (4-check) ->
ghi `data/interim/labels.parquet` + `_label_log.json`. Exit 0 nếu OK, 1 nếu lỗi.

    python scripts/label_phase1.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import label as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
SRC = INTERIM / "integrated.parquet"
OUT = INTERIM / "labels.parquet"
LOG = INTERIM / "_label_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def main() -> int:
    try:
        panel = pd.read_parquet(SRC)

        labels = L.build_labels(panel)
        L.validate(labels, panel)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(OUT, index=False)

        ties = L.tie_counts(panel)
        tail_nan = {f"y_{k}": int(labels[f"y_{k}"].isna().sum()) for k in L.HORIZONS}
        pct_pos = {
            f"y_{k}": round(
                100 * float((labels[f"y_{k}"] == 1.0).sum())
                / int(labels[f"y_{k}"].notna().sum()), 2
            )
            for k in L.HORIZONS
        }

        log = {
            "status": "ok",
            "output": str(OUT),
            "rows": len(labels),
            "labels": L.LABELS,
            "date_min": str(labels["date"].min().date()),
            "date_max": str(labels["date"].max().date()),
            "tie_counts": {str(k): v for k, v in ties.items()},
            "tail_nan": tail_nan,
            "pct_pos": pct_pos,
            "checks": ["spine_aligned", "label_domain",
                       "tail_nan_exact", "tie_convention"],
            "generated_at": datetime.now(TZ).isoformat(),
        }
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {len(labels)} phiên x {len(L.LABELS)} nhãn -> {OUT}")
        print(f"     {log['date_min']} -> {log['date_max']}")
        print(f"     tie_counts: {ties}")
        print(f"     pct_pos(+1): {pct_pos}")
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