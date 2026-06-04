"""scripts/baselines_phase1.py — runner Step 9 (baselines trên val + test).

Đọc `data/processed/features.parquet` -> build 3 baseline trên val+test mỗi k ->
validate -> ghi `data/processed/predictions_baseline.parquet` + `_baselines_log.json`
(kèm accuracy từng (k,segment) để đối chiếu nhanh). Exit 0 nếu OK, 1 nếu lỗi.

    python scripts/baselines_phase1.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import baselines as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "features.parquet"
OUT = ROOT / "data" / "processed" / "predictions_baseline.parquet"
LOG = ROOT / "data" / "processed" / "_baselines_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _acc(pred: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(y)
    return round(float((pred[m] == y[m]).mean()) * 100, 2) if m.any() else None


def main() -> int:
    try:
        df = pd.read_parquet(SRC)
        out = B.build_baselines(df)
        B.validate(out, df)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)

        # accuracy từng (k, segment) cho từng baseline
        acc = {}
        for k in B.HORIZONS:
            acc[str(k)] = {}
            for seg in ("val", "test"):
                g = out[(out["k"] == k) & (out["segment"] == seg)]
                y = g["y_true"].to_numpy(dtype=float)
                acc[str(k)][seg] = {
                    "n_label": int((~np.isnan(y)).sum()),
                    **{c: _acc(g[c].to_numpy(dtype=float), y) for c in B.PRED_COLS},
                }

        dyn_eq_always = bool((out["dyn_majority"] == out["always_pos"]).all())
        log = {
            "status": "ok",
            "output": str(OUT),
            "rows": len(out),
            "segments": ["val", "test"],
            "baselines": list(B.PRED_COLS),
            "accuracy_pct": acc,
            "dyn_majority_eq_always_pos": dyn_eq_always,
            "generated_at": datetime.now(TZ).isoformat(),
        }
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[OK] baselines -> {OUT.name} ({len(out)} dòng, val+test x {len(B.HORIZONS)} horizon)")
        print(f"     dyn_majority == always_pos: {dyn_eq_always}")
        print(f"     accuracy % (nhãn dùng được):")
        for k in B.HORIZONS:
            for seg in ("val", "test"):
                a = acc[str(k)][seg]
                print(f"       k={k:<2d} {seg:<4s} n={a['n_label']:<4d}  "
                      f"persistence={a['persistence']}  "
                      f"dyn_majority={a['dyn_majority']}  always_pos={a['always_pos']}")
        return 0

    except Exception as e:  # noqa: BLE001
        err = {"status": "error", "error": f"{type(e).__name__}: {e}",
               "generated_at": datetime.now(TZ).isoformat()}
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ERROR] {err['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())