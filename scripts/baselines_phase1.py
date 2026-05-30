"""Runner Step 9: features.parquet -> predictions_baseline.parquet (+ _baselines_log.json).

Chạy từ gốc repo: python scripts/baselines_phase1.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo lên sys.path

import numpy as np
import pandas as pd

from src.model.baselines import build_baselines, validate, HORIZONS, PRED_COLS
from src.model.walk_forward import walk_forward_splits

IN = Path("data/processed/features.parquet")
OUT = Path("data/processed/predictions_baseline.parquet")
LOG = Path("data/processed/_baselines_log.json")


def _acc(pred: pd.Series, y_true: pd.Series):
    """Accuracy bỏ qua dòng y_true NaN (đuôi)."""
    m = y_true.notna().to_numpy()
    if m.sum() == 0:
        return None
    return round(float((pred.to_numpy()[m] == y_true.to_numpy()[m]).mean()), 4)


def main() -> int:
    try:
        df = pd.read_parquet(IN)
        out = build_baselines(df)
        validate(out, df)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)

        log = {"status": "ok", "rows": int(len(out)), "per_k": {}}
        for k in HORIZONS:
            sub = out[out["k"] == k]
            log["per_k"][str(k)] = {
                "rows": int(len(sub)),
                "weeks": sum(1 for _ in walk_forward_splits(df["date"], k)),
                "y_true_nan": int(sub["y_true"].isna().sum()),
                "acc": {c: _acc(sub[c], sub["y_true"]) for c in PRED_COLS},
                "dyn_majority_pos_frac": round(float(np.mean(sub["dyn_majority"] == 1.0)), 4),
            }
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))

        print(f"[ok] {len(out)} dòng -> {OUT}")
        for k in HORIZONS:
            p = log["per_k"][str(k)]
            a = p["acc"]
            print(f"  k={k:<2d}: {p['rows']:4d} dòng, {p['weeks']:3d} tuần | "
                  f"acc pers/dynmaj/+1 = {a['persistence']}/{a['dyn_majority']}/{a['always_pos']}")
        return 0
    except Exception as e:  # log lỗi, exit 1 (tiện Phase 2)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2))
        print(f"[error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())