"""Runner Step 11: features.parquet + hparams.json -> predictions_model.parquet (+ log).

Chạy từ gốc repo:
    python scripts/train_infer_phase1.py                      # FULL: 3 model × 4 horizon
    python scripts/train_infer_phase1.py --models elastic_net lightgbm   # bỏ LSTM (không cần GPU)
    python scripts/train_infer_phase1.py --horizons 1 --models lightgbm  # smoke test plumbing

FULL = đủ 4 horizon × 3 model → ghi artifact canonical + validate + log.
Smoke (thiếu horizon hoặc model) → ghi _smoke_predictions_model.parquet, KHÔNG đụng
file canonical, không validate chặt (để bảo vệ artifact thật).
"""
import argparse
import json
import sys
from pathlib import Path

# Nạp torch ĐẦU TIÊN (trước numpy/pandas/lightgbm/sklearn). Trên Windows, nạp torch SAU
# các lib kéo MKL/OpenMP làm c10.dll init fail (WinError 1114). try/except để EN/LGB vẫn
# chạy khi torch vắng/hỏng.
try:
    import torch  # noqa: F401
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo lên sys.path

import numpy as np
import pandas as pd

from src.model.train_infer import (
    CANONICAL, HORIZONS, MODELS, PROBA_COLS, run, validate,
)
from model.split import walk_forward_splits

FEATURES = Path("data/processed/features.parquet")
HPARAMS = Path("config/hparams.json")
BASELINE = Path("data/processed/predictions_baseline.parquet")
OUT = Path("data/processed/predictions_model.parquet")
OUT_SMOKE = Path("data/processed/_smoke_predictions_model.parquet")
LOG = Path("data/processed/_train_infer_log.json")

# proba col ↔ model name (để log accuracy đúng cột)
COL_OF = {"elastic_net": "en_proba", "lightgbm": "lgb_proba", "lstm": "lstm_proba"}


def _read_json(path: Path) -> dict:
    """Đọc JSON bền encoding: hparams.json do Step 10 ghi bằng cp1252 (Windows) nên
    ép utf-8 sẽ vỡ ở ký tự '§'. Thử utf-8 → cp1252 → latin-1."""
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Không decode được {path}")


def _acc(proba: pd.Series, y_true: pd.Series):
    """0-1 accuracy: sign(proba≥0.5)→+1, bỏ qua dòng y_true NaN (đuôi)."""
    m = y_true.notna().to_numpy() & np.isfinite(proba.to_numpy())
    if m.sum() == 0:
        return None
    pred = np.where(proba.to_numpy()[m] >= 0.5, 1.0, -1.0)
    return round(float((pred == y_true.to_numpy()[m]).mean()), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS), choices=HORIZONS)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--gpu", type=int, default=0, help="index card GPU cho LSTM (mặc định 0)")
    args = ap.parse_args()
    horizons, models = sorted(set(args.horizons)), [m for m in MODELS if m in set(args.models)]
    full = set(horizons) == set(HORIZONS) and set(models) == set(MODELS)

    try:
        df = pd.read_parquet(FEATURES)
        hparams = _read_json(HPARAMS)
        if "lstm" in models:
            print(f"[gpu] LSTM dùng cuda:{args.gpu} = {torch.cuda.get_device_name(args.gpu)}")
        out = run(df, hparams, horizons=horizons, models=models, gpu=args.gpu)

        dst = OUT if full else OUT_SMOKE
        dst.parent.mkdir(parents=True, exist_ok=True)

        if full:
            baseline = pd.read_parquet(BASELINE)
            validate(out, baseline)

        out.to_parquet(dst, index=False)

        # log per (model, k): acc, zero_one_loss, n_weeks, n_rows
        log = {"status": "ok", "mode": "full" if full else "smoke",
               "horizons": horizons, "models": models,
               "rows": int(len(out)), "output": str(dst), "per_k": {}}
        for k in horizons:
            sub = out[out["k"] == k]
            wk = sum(1 for _ in walk_forward_splits(df["date"], k))
            entry = {"rows": int(len(sub)), "weeks": wk,
                     "y_true_nan": int(sub["y_true"].isna().sum()), "model": {}}
            for m in models:
                a = _acc(sub[COL_OF[m]], sub["y_true"])
                entry["model"][m] = {"acc": a, "zero_one_loss": None if a is None else round(1 - a, 4)}
            log["per_k"][str(k)] = entry
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

        tag = "FULL" if full else "SMOKE"
        print(f"[ok/{tag}] {len(out)} dòng -> {dst}")
        for k in horizons:
            e = log["per_k"][str(k)]
            accs = " ".join(f"{m}={e['model'][m]['acc']}" for m in models)
            print(f"  k={k:<2d}: {e['rows']:4d} dòng, {e['weeks']:3d} tuần | acc {accs}")
        if not full:
            print("  [smoke] KHÔNG ghi đè predictions_model.parquet, không validate chặt.")
        return 0
    except Exception as e:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())