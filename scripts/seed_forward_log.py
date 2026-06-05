"""Seed sổ forward — backfill HONEST đoạn live SAU khi model freeze (chạy MỘT LẦN).

Deploy là refit-all trên data tĩnh Phase 1 (data tới hết phiên 2026-05-28). Một phiên t
là OOS cho horizon k khi nhãn của nó cần giá NGOÀI phạm vi train, tức t+k > 2026-05-28.
=> mốc bắt đầu OOS THEO TỪNG k: phiên thứ (k-1) trước 2026-05-28.
  k=1 → từ 2026-05-28 · k=5 → ~5 phiên trước · k=10 · k=20 (xa nhất).
Backfill đúng các phiên này (không sớm hơn → tránh in-sample), khớp liền với phần "past"
(fit-train OOS) vốn dừng ngay trước mốc đó. Chạy một lần; idempotent nếu chạy lại.

    python scripts/seed_forward_log.py [--end YYYY-MM-DD]

Rebuild + publish sau đó:
    python scripts/web_build_phase1.py && python scripts/web_app_phase1.py && python scripts/publish_phase3.py
"""
from __future__ import annotations

# Windows: torch TRƯỚC numpy/pandas (LSTM k=5).
try:
    import torch  # noqa: F401
except Exception:
    pass

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import forward_log as FL   # noqa: E402
from src.web import inference as W        # noqa: E402

PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
DEPLOY = ROOT / "deploy"
FWLOG_PATH = ROOT / "data" / "forward_log.parquet"
HORIZONS = (1, 5, 10, 20)
FROZEN_LAST = pd.Timestamp("2026-05-28")   # phiên cuối của data Phase 1 (model freeze)


def _predict_one(feats_d: pd.DataFrame, k: int, entry: dict, p_at: dict):
    d = feats_d["date"].iloc[-1].normalize()
    try:
        fwd = W._predict_forward(feats_d, k, entry, DEPLOY)
    except Exception as e:
        print(f"    ⚠ k={k} @ {d.date()} lỗi: {type(e).__name__}: {e}")
        return None
    return {"from_date": d, "k": k, "run_date": d, "model": entry["model"],
            "feature_set": entry["feature_set"], "proba": fwd["proba"],
            "pred": fwd["pred"], "from_price": p_at.get(d, float("nan"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="phiên gốc cuối (mặc định = phiên mới nhất)")
    args = ap.parse_args()

    features = pd.read_parquet(PROC / "features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    price = (pd.read_parquet(RAW / "tcb_price.parquet")[["date", "close"]]
             .rename(columns={"close": "adj_close"}))
    price["date"] = pd.to_datetime(price["date"])
    p_at = dict(zip(price["date"].dt.normalize(), price["adj_close"]))
    manifest = json.loads((DEPLOY / "manifest.json").read_text(encoding="utf-8"))

    feat_dates = list(features["date"])
    end = pd.Timestamp(args.end).normalize() if args.end else feat_dates[-1]
    le = [d for d in feat_dates if d <= FROZEN_LAST]
    if not le:
        print(f"✗ không thấy phiên <= {FROZEN_LAST.date()} trong features."); return 1
    idx_f = feat_dates.index(le[-1])   # vị trí phiên freeze trong lịch

    rows = []
    print(f"Seed honest theo từng k (mốc freeze = {feat_dates[idx_f].date()}):")
    for k in HORIZONS:
        entry = manifest[str(k)]
        start_idx = max(0, idx_f - (k - 1))
        window = [d for d in feat_dates[start_idx:] if d <= end]
        if not window:
            continue
        print(f"  k={k:>2}: {window[0].date()} .. {window[-1].date()}  ({len(window)} phiên OOS)")
        for d in window:
            r = _predict_one(features[features["date"] <= d], k, entry, p_at)
            if r:
                rows.append(r)

    new = pd.DataFrame(rows)
    if new.empty:
        print("✗ không sinh được record nào."); return 1

    log = pd.read_parquet(FWLOG_PATH) if FWLOG_PATH.exists() else FL.empty_log()
    log = FL.append_predictions(log, new)
    log = FL.resolve(log, price, price_col="adj_close")
    FWLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(FWLOG_PATH, index=False)

    print(f"\n→ sổ forward: {FWLOG_PATH}  ({len(log)} dòng)")
    print(FL.summary(log).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())