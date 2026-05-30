"""Runner Step 10: features.parquet (Phase-0) -> config/hparams.json (+ _tune_log.json).

Tune Elastic Net / LightGBM MỘT LẦN trên Phase-0, khóa lại. LSTM frozen (§5.4).
CPU-only (không GPU). Chạy từ gốc repo: python scripts/tune_phase1.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo lên sys.path

import pandas as pd

from src.model.tuning import HORIZONS, tune_all, validate

IN = Path("data/processed/features.parquet")
OUT = Path("config/hparams.json")
LOG = Path("config/_tune_log.json")


def main() -> int:
    try:
        df = pd.read_parquet(IN)
        hp = tune_all(df)
        validate(hp, df)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(hp, ensure_ascii=False, indent=2))

        log = {"status": "ok", "meta": hp["meta"], "per_k": {}}
        for k in HORIZONS:
            en = hp["elastic_net"]["per_horizon"][str(k)]
            gb = hp["lightgbm"]["per_horizon"][str(k)]
            log["per_k"][str(k)] = {
                "n_train": en["n_train"],
                "en_lambda": round(en["lambda"], 6),
                "en_lambda_at_grid_edge": en["at_grid_edge"],
                "en_val_logloss": round(en["val_logloss"], 4),
                "lgb_n_estimators": gb["n_estimators"],
                "lgb_val_logloss": round(gb["val_logloss"], 4),
            }
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))

        print(f"[ok] hparams -> {OUT}")
        print(f"  Phase-0: {hp['meta']['phase0_start']} -> {hp['meta']['phase0_end']} "
              f"| val {hp['meta']['val_size']} phiên (15%)")
        for k in HORIZONS:
            p = log["per_k"][str(k)]
            edge = "  ⚠ λ chạm biên grid" if p["en_lambda_at_grid_edge"] else ""
            print(f"  k={k:<2d}: n_train={p['n_train']:3d} | "
                  f"EN λ={p['en_lambda']:.4f} ll={p['en_val_logloss']} | "
                  f"LGB n_est={p['lgb_n_estimators']:<4d} ll={p['lgb_val_logloss']}{edge}")
        return 0
    except Exception as e:  # log lỗi, exit 1 (tiện Phase 2)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2))
        print(f"[error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())