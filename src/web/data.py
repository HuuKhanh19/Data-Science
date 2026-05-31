"""Lớp truy cập dữ liệu cho web (Step 13) — Phase 2/3 tái dùng.

Thuần đọc artifact đã sinh (predictions_model.parquet, results.json, results_L1.json),
không tính toán khoa học (đã làm ở Step 9–12). API chỉ là lớp mỏng phía trên module này.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PRED_MODEL = PROC / "predictions_model.parquet"
RESULTS = REPORTS / "results.json"
RESULTS_L1 = REPORTS / "results_L1.json"

HORIZONS = (1, 5, 10, 20)
MODEL_COL = {"en": "en_proba", "lgb": "lgb_proba", "lstm": "lstm_proba"}

CAVEAT = ("Các model KHÔNG có edge thống kê có ý nghĩa so với baseline (xem phần "
          "Predictability). Dự đoán hiển thị để minh bạch khoa học, KHÔNG phải khuyến "
          "nghị giao dịch.")

# Phase 1 = snapshot tĩnh; Phase 2 (pipeline tự động) đổi thành "live" + trỏ
# latest_inference()/inference_history() vào live store. API/UI giữ nguyên.
MODE = "snapshot"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_results() -> dict | None:
    """results.json (study chính, 20 feature)."""
    return _read_json(RESULTS)


def load_results_l1() -> dict | None:
    """results_L1.json (ablation L1-only); None nếu chưa chạy thí nghiệm."""
    return _read_json(RESULTS_L1)


def latest_inference() -> dict:
    """Dự đoán cho phiên mới nhất (mỗi horizon × 3 model).

    Lấy dòng có date lớn nhất của từng k trong predictions_model.parquet — đây là dự đoán
    live (y_true còn NaN vì cửa sổ [t, t+k] chưa hiện thực hóa). sign = proba≥0.5 → UP.
    """
    df = pd.read_parquet(PRED_MODEL)          # FileNotFoundError nếu chưa có → API bắt
    df["date"] = pd.to_datetime(df["date"])
    as_of = df["date"].max()

    horizons = []
    for k in HORIZONS:
        sub = df[df["k"] == k]
        row = sub.loc[sub["date"].idxmax()]
        models = {}
        for name, col in MODEL_COL.items():
            p = float(row[col])
            models[name] = {"proba_up": round(p, 4),
                            "direction": "UP" if p >= 0.5 else "DOWN"}
        horizons.append({
            "k": int(k),
            "date": str(pd.Timestamp(row["date"]).date()),
            "realized": bool(pd.notna(row["y_true"])),
            "models": models,
        })
    return {"as_of": str(pd.Timestamp(as_of).date()), "mode": MODE, "caveat": CAVEAT,
            "horizons": horizons}


def inference_history(k: int, n: int = 20) -> dict:
    """N phiên ĐÃ hiện thực hóa gần nhất của horizon k: dự đoán vs thực tế + hit-rate.

    Chỉ lấy dòng y_true không NaN (đã biết kết quả). hit-rate kèm base_rate cùng cửa sổ
    để thấy ngay 'không có edge' khi hit-rate ≈ base_rate (study negative).
    """
    df = pd.read_parquet(PRED_MODEL)
    df["date"] = pd.to_datetime(df["date"])
    sub = df[(df["k"] == k) & df["y_true"].notna()].sort_values("date").tail(n)
    n_real = len(sub)

    hits = {m: 0 for m in MODEL_COL}
    rows = []
    for _, r in sub.iterrows():
        actual = int(r["y_true"])
        entry = {"date": str(pd.Timestamp(r["date"]).date()),
                 "actual": "UP" if actual == 1 else "DOWN", "models": {}}
        for name, col in MODEL_COL.items():
            pred = 1 if float(r[col]) >= 0.5 else -1
            hit = pred == actual
            hits[name] += int(hit)
            entry["models"][name] = {"direction": "UP" if pred == 1 else "DOWN", "hit": hit}
        rows.append(entry)

    base_rate = round(float((sub["y_true"] == 1).mean()), 4) if n_real else None
    hit_rate = {m: (round(hits[m] / n_real, 4) if n_real else None) for m in MODEL_COL}
    return {"k": int(k), "n": n_real, "base_rate": base_rate,
            "hit_rate": hit_rate, "rows": rows}