"""scripts/split_phase1.py — runner Step 7 (định nghĩa chỉ số split 80:10:10 + embargo).

Đọc `data/processed/features.parquet` -> validate split -> in bảng tóm tắt + ghi
`data/processed/_split_log.json`. KHÔNG xuất parquet: split chỉ là CHỈ SỐ, việc áp
dụng (fit Z-score trên train, chấm val/test) nằm ở Step 11. Exit 0 nếu OK, 1 nếu lỗi.

    python scripts/split_phase1.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import split as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "features.parquet"
LOG = ROOT / "data" / "processed" / "_split_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def main() -> int:
    try:
        df = pd.read_parquet(SRC)
        S.validate(df)

        per_k = {str(k): S.split_summary(df, k) for k in S.HORIZONS}
        dates = pd.to_datetime(df["date"])
        sp1 = S.make_split(df["date"], 1)  # ranh giới k-độc lập

        log = {
            "status": "ok",
            "n_rows": len(df),
            "date_min": str(dates.iloc[0].date()),
            "date_max": str(dates.iloc[-1].date()),
            "fractions": {
                "train": S.TRAIN_FRAC,
                "val": S.VAL_FRAC,
                "test": round(1.0 - S.TRAIN_FRAC - S.VAL_FRAC, 4),
            },
            "boundary_train_val": {
                "pos": int(sp1.train_end),
                "date": str(dates.iloc[sp1.train_end].date()),
            },
            "boundary_val_test": {
                "pos": int(sp1.val_end),
                "date": str(dates.iloc[sp1.val_end].date()),
            },
            "per_k": per_k,
            "note": "split chỉ là chỉ số; áp dụng (z-score fit-train, chấm val/test) ở Step 11",
            "generated_at": datetime.now(TZ).isoformat(),
        }
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── bảng ra màn hình ──
        print(f"[OK] features.parquet: {len(df)} phiên | {log['date_min']} -> {log['date_max']}")
        print(f"     tỉ lệ train/val/test = "
              f"{log['fractions']['train']}/{log['fractions']['val']}/{log['fractions']['test']}")
        print(f"     ranh giới train|val @ idx {sp1.train_end} ({log['boundary_train_val']['date']})"
              f"  |  val|test @ idx {sp1.val_end} ({log['boundary_val_test']['date']})")
        print()
        for k in S.HORIZONS:
            s = per_k[str(k)]
            tr, va, te = s["train"], s["val"], s["test"]
            print(f"  k={k:<2d} | train {tr['n']:<5d} [{tr['date_start']}..{tr['date_end']}] "
                  f"pos {tr['pct_pos']}%")
            print(f"        | val   {va['n']:<5d} [{va['date_start']}..{va['date_end']}] "
                  f"pos {va['pct_pos']}%")
            print(f"        | test  {te['n']:<5d} [{te['date_start']}..{te['date_end']}] "
                  f"pos {te['pct_pos']}%  -> nhãn dùng được {te['n_label']} "
                  f"(độc lập ~{te['n_independent']})")
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