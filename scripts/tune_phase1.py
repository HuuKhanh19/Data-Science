"""scripts/tune_phase1.py — runner Step 10 (tune EN+LGB trên val, theo feature_set).

Đọc `data/processed/features.parquet` + `config/feature_sets.json` -> tune mỗi
(feature_set, horizon) -> validate -> ghi `config/hparams.json` + `config/_tune_log.json`.
LSTM frozen-config được ghi kèm (không tune). Exit 0 nếu OK, 1 nếu lỗi.

CHỌN (feature_set × model) deploy KHÔNG ở đây — thực hiện ở Step 11 (cần val-score
của cả LSTM, fit một lần). Đây chỉ khóa hyperparam.

    python scripts/tune_phase1.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import tuning as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "features.parquet"
FSETS = ROOT / "config" / "feature_sets.json"
OUT = ROOT / "config" / "hparams.json"
LOG = ROOT / "config" / "_tune_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def main() -> int:
    try:
        df = pd.read_parquet(SRC)
        feature_sets = json.loads(FSETS.read_text(encoding="utf-8"))
        feature_sets = {k: feature_sets[k] for k in T.FEATURE_SET_ORDER}  # bỏ _meta

        hp = T.tune_all(df, feature_sets, verbose=True)
        T.validate(hp, feature_sets)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(hp, ensure_ascii=False, indent=2), encoding="utf-8")

        # log gọn: val logloss/MCC tuned mỗi (fs, k, model)
        summary = {}
        for fs in T.FEATURE_SET_ORDER:
            summary[fs] = {}
            for k in T.HORIZONS:
                e = hp["feature_sets"][fs]["per_k"][str(k)]
                summary[fs][str(k)] = {
                    "en": {"lambda": round(e["elastic_net"]["lambda"], 5),
                           "val_logloss": round(e["elastic_net"]["val"]["logloss"], 4),
                           "val_mcc": round(e["elastic_net"]["val"]["mcc"], 4),
                           "val_balacc": round(e["elastic_net"]["val"]["balacc"], 4),
                           "pred_pos_rate": round(e["elastic_net"]["val"]["pred_pos_rate"], 3)},
                    "lgb": {"n_est": e["lightgbm"]["n_estimators"],
                            "val_logloss": round(e["lightgbm"]["val"]["logloss"], 4),
                            "val_mcc": round(e["lightgbm"]["val"]["mcc"], 4),
                            "val_balacc": round(e["lightgbm"]["val"]["balacc"], 4),
                            "pred_pos_rate": round(e["lightgbm"]["val"]["pred_pos_rate"], 3)},
                }
        log = {"status": "ok", "output": str(OUT), "feature_sets": list(T.FEATURE_SET_ORDER),
               "tuned_val_summary": summary, "generated_at": datetime.now(TZ).isoformat()}
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[OK] tune -> {OUT.name}")
        print(f"     val MCC / balacc tuned (≥0 mới có chút skill; chú ý pred_pos_rate≈1 = thoái hoá):")
        for fs in T.FEATURE_SET_ORDER:
            print(f"  [{fs}]")
            for k in T.HORIZONS:
                s = summary[fs][str(k)]
                print(f"    k={k:<2d} EN  mcc={s['en']['val_mcc']:+.3f} balacc={s['en']['val_balacc']:.3f} "
                      f"pos={s['en']['pred_pos_rate']:.2f} ll={s['en']['val_logloss']:.4f} | "
                      f"LGB mcc={s['lgb']['val_mcc']:+.3f} balacc={s['lgb']['val_balacc']:.3f} "
                      f"pos={s['lgb']['pred_pos_rate']:.2f} ll={s['lgb']['val_logloss']:.4f}")
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