"""Seed sổ forward — backfill HONEST từ 2026-05-28 tới phiên mới nhất (chạy MỘT LẦN).

Vì deploy là refit-all trên data tĩnh P1 (rows có nhãn, tới hết 2026-05-27): MỌI phiên
>= 2026-05-28 đều NẰM NGOÀI train (nhãn đuôi NaN khi freeze) -> dự đoán forward từ các
phiên này là OOS thật, hợp lệ để đưa vào track record. KHÔNG backfill trước 28-05 vì
với k=1 các phiên đó đã nằm trong train (in-sample -> không trung thực).

Sau seed, các forward đã đủ k phiên (vd 28-05 +1/+5) tự được chấm bằng giá tới nay.
Self-contained: không import script khác.

    python scripts/seed_forward_log.py [--start 2026-05-28] [--end YYYY-MM-DD]

Rebuild web sau đó:  python scripts/web_build_phase1.py && python scripts/web_app_phase1.py
"""
from __future__ import annotations

# Windows: torch TRƯỚC numpy/pandas (LSTM k=5 cho forward).
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


def _records_asof(feats: pd.DataFrame, manifest: dict, p_at: dict, deploy_dir: Path) -> list[dict]:
    """Dự đoán forward as-of phiên cuối của `feats` (mỗi horizon một dòng)."""
    last_date = feats["date"].iloc[-1].normalize()
    from_price = p_at.get(last_date, float("nan"))
    rows = []
    for k in HORIZONS:
        entry = manifest.get(str(k))
        if entry is None:
            continue
        try:
            fwd = W._predict_forward(feats, k, entry, deploy_dir)
        except Exception as e:
            print(f"  ⚠ forward k={k} @ {last_date.date()} lỗi: {type(e).__name__}: {e}")
            continue
        rows.append({
            "from_date": last_date, "k": k, "run_date": last_date,
            "model": entry["model"], "feature_set": entry["feature_set"],
            "proba": fwd["proba"], "pred": fwd["pred"], "from_price": from_price,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-28")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    features = pd.read_parquet(PROC / "features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    price = (pd.read_parquet(RAW / "tcb_price.parquet")[["date", "close"]]
             .rename(columns={"close": "adj_close"}))
    price["date"] = pd.to_datetime(price["date"])
    p_at = dict(zip(price["date"].dt.normalize(), price["adj_close"]))
    manifest = json.loads((DEPLOY / "manifest.json").read_text(encoding="utf-8"))

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize() if args.end else features["date"].max()
    window = features.loc[(features["date"] >= start) & (features["date"] <= end), "date"].tolist()
    if not window:
        print(f"✗ không có phiên nào trong [{start.date()} .. {end.date()}]")
        return 1
    print(f"Seed {len(window)} phiên gốc: {window[0].date()} .. {window[-1].date()} "
          f"(OOS thật — deploy không train trên các phiên này)")

    rows = []
    for d in window:
        rows += _records_asof(features[features["date"] <= d], manifest, p_at, DEPLOY)
    new = pd.DataFrame(rows)
    if new.empty:
        print("✗ không sinh được record nào.")
        return 1

    log = pd.read_parquet(FWLOG_PATH) if FWLOG_PATH.exists() else FL.empty_log()
    log = FL.append_predictions(log, new)
    log = FL.resolve(log, price, price_col="adj_close")
    FWLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(FWLOG_PATH, index=False)

    print(f"\n→ sổ forward: {FWLOG_PATH}  ({len(log)} dòng)")
    print(FL.summary(log).to_string(index=False))
    print("\nRebuild web:  python scripts/web_build_phase1.py && python scripts/web_app_phase1.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())