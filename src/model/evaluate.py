"""Step 12 (ĐV6) — Evaluate (TCB direction, Khung A). Logic thuần, không I/O.

THAY bản confirmatory cũ. Khung A đã BỎ: block bootstrap, Diebold-Mariano,
Holm-Bonferroni, verdict §8, analytical-50%, pre-reg. Ở đây chỉ còn:

  - metric TEST (chạm 1 lần) toàn lưới (feature_set × model × k):
      accuracy, balanced-acc, MCC, AUC, precision/recall/F1 từng lớp, confusion.
  - baselines (persistence / dyn_majority / always_pos) chấm trên cùng test.
  - chẩn đoán DEGENERATE (auto-+1): pred_pos_rate vs base_pos_rate; cờ degenerate
    nếu pred_pos_rate ≥ 0.97 & balacc ≈ 0.5 & |MCC| ≈ 0 → thoái hoá về majority.
  - Δ vs baselines (acc/balacc/mcc), tách riêng config DEPLOY (best-single ĐV5).

Quy ước: sign = (proba ≥ 0.5 → +1, else −1); +1 = positive class (khớp baseline,
tie→+1). Chỉ chấm segment=test, bỏ đuôi y_true NaN (inference live). val KHÔNG chấm
ở đây (đã dùng để CHỌN ở ĐV5). accuracy thô CHỈ để tham khảo — phán quyết dựa
balanced-acc / MCC / Δ-vs-always_pos vì test imbalance (base_pos_rate ~48–52%, lật ở k=20).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HORIZONS = (1, 5, 10, 20)
MODELS = {"en": "en_proba", "lgb": "lgb_proba", "lstm": "lstm_proba"}
MODEL_FULL = {"en": "elastic_net", "lgb": "lightgbm", "lstm": "lstm"}
FULL_TO_SHORT = {v: k for k, v in MODEL_FULL.items()}
BASELINES = ("persistence", "dyn_majority", "always_pos")
FEATURE_SET_ORDER = ("l1", "eda", "full")

# Cờ degenerate (data.md §Chẩn đoán): auto-+1 ⇔ gần như không bao giờ đoán −1.
DEGEN_POS = 0.97        # pred_pos_rate ≥ → "luôn +1"
DEGEN_BALACC = 0.53     # balanced-acc ≈ 0.5
DEGEN_MCC = 0.05        # |MCC| ≈ 0


def _sign_from_proba(p: np.ndarray) -> np.ndarray:
    """proba ≥ 0.5 → +1, else −1 (tie 0.5 → +1, khớp baseline)."""
    return np.where(np.asarray(p, float) >= 0.5, 1, -1).astype(np.int64)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> dict:
    """Metric đầy đủ cho 1 cặp (y_true, y_pred). +1 = positive. proba (tuỳ chọn) → AUC.

    y_true, y_pred ∈ {−1,+1}. Trả acc/balacc/mcc/auc + confusion + precision/recall/F1
    từng lớp + pred_pos_rate.
    """
    y_true = np.asarray(y_true, np.int64)
    y_pred = np.asarray(y_pred, np.int64)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == -1) & (y_true == -1)))
    fp = int(np.sum((y_pred == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else np.nan
    tpr = tp / (tp + fn) if (tp + fn) else np.nan        # recall pos
    tnr = tn / (tn + fp) if (tn + fp) else np.nan        # recall neg
    balacc = float(np.nanmean([tpr, tnr]))
    prec_pos = tp / (tp + fp) if (tp + fp) else np.nan
    prec_neg = tn / (tn + fn) if (tn + fn) else np.nan

    def _f1(p, r):
        return (2 * p * r / (p + r)) if (np.isfinite(p) and np.isfinite(r) and (p + r) > 0) else np.nan

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    auc = np.nan
    if proba is not None:
        proba = np.asarray(proba, float)
        if len(np.unique(y_true)) == 2 and np.all(np.isfinite(proba)):
            auc = float(roc_auc_score((y_true == 1).astype(int), proba))

    return {
        "n": n, "acc": float(acc), "balacc": balacc, "mcc": float(mcc),
        "auc": (float(auc) if np.isfinite(auc) else None),
        "pred_pos_rate": float((y_pred == 1).mean()) if n else float("nan"),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": {"pos": float(prec_pos), "neg": float(prec_neg)},
        "recall": {"pos": float(tpr), "neg": float(tnr)},
        "f1": {"pos": float(_f1(prec_pos, tpr)), "neg": float(_f1(prec_neg, tnr))},
    }


def _is_degenerate(m: dict) -> bool:
    """auto-+1: gần như chỉ đoán +1 & không phân biệt (balacc≈0.5 & |MCC|≈0)."""
    return (m["pred_pos_rate"] >= DEGEN_POS
            and m["balacc"] <= DEGEN_BALACC
            and abs(m["mcc"]) <= DEGEN_MCC)


def _deltas(m: dict, base_m: dict) -> dict:
    """Δ model − baseline cho acc/balacc/mcc."""
    return {
        "acc": round(m["acc"] - base_m["acc"], 4),
        "balacc": round(m["balacc"] - base_m["balacc"], 4),
        "mcc": round(m["mcc"] - base_m["mcc"], 4),
    }


def _test_frames(model_df: pd.DataFrame, base_df: pd.DataFrame):
    """Lọc segment=test, ép date, sanity cột."""
    need_m = {"date", "k", "feature_set", "segment", "y_true", *MODELS.values()}
    need_b = {"date", "k", "segment", "y_true", *BASELINES}
    if not need_m.issubset(model_df.columns):
        raise ValueError(f"[evaluate] predictions_model thiếu cột: {need_m - set(model_df.columns)}")
    if not need_b.issubset(base_df.columns):
        raise ValueError(f"[evaluate] predictions_baseline thiếu cột: {need_b - set(base_df.columns)}")
    m = model_df[model_df["segment"] == "test"].copy()
    b = base_df[base_df["segment"] == "test"].copy()
    m["date"] = pd.to_datetime(m["date"])
    b["date"] = pd.to_datetime(b["date"])
    return m, b


def build_results(model_df: pd.DataFrame, base_df: pd.DataFrame,
                  deploy_choice: dict | None = None) -> dict:
    """Chấm test toàn lưới + baselines + deploy. Trả dict results.json-ready."""
    m_test, b_test = _test_frames(model_df, base_df)
    fsets_present = [fs for fs in FEATURE_SET_ORDER if fs in set(m_test["feature_set"])]

    res = {
        "meta": {
            "scheme": "80:10:10 + embargo k; chấm TEST 1 lần; sign=proba≥0.5→+1",
            "note": "accuracy thô tham khảo; phán quyết = balanced-acc / MCC / Δ-vs-always_pos",
            "degen_rule": {"pred_pos_rate>=": DEGEN_POS, "balacc<=": DEGEN_BALACC, "|mcc|<=": DEGEN_MCC},
            "feature_sets": fsets_present,
            "models": list(MODEL_FULL.values()),
            "baselines": list(BASELINES),
        },
        "horizons": {},
    }

    for k in HORIZONS:
        bk = b_test[b_test["k"] == k].dropna(subset=["y_true"]).copy()
        if bk.empty:
            continue
        bk = bk.sort_values("date").reset_index(drop=True)
        yb = bk["y_true"].to_numpy(float)
        y_true = np.where(yb > 0, 1, -1).astype(np.int64)
        base_pos_rate = float((y_true == 1).mean())

        base_out = {}
        for bcol in BASELINES:
            bp = np.where(bk[bcol].to_numpy(float) > 0, 1, -1).astype(np.int64)
            base_out[bcol] = metrics(y_true, bp)        # baseline không có proba → AUC null

        always = base_out["always_pos"]
        grid = {}
        for fs in fsets_present:
            mk = m_test[(m_test["k"] == k) & (m_test["feature_set"] == fs)].copy()
            mk = mk.merge(bk[["date"]], on="date", how="inner").sort_values("date")
            if len(mk) != len(bk):
                raise ValueError(f"[evaluate] k={k} fs={fs}: test lệch baseline "
                                 f"({len(mk)} vs {len(bk)})")
            grid[fs] = {}
            for short, col in MODELS.items():
                proba = mk[col].to_numpy(float)
                if np.all(np.isnan(proba)):
                    continue                            # model không chạy
                pred = _sign_from_proba(proba)
                mt = metrics(y_true, pred, proba)
                mt["degenerate"] = _is_degenerate(mt)
                mt["delta_vs_always_pos"] = _deltas(mt, always)
                mt["delta_vs_persistence"] = _deltas(mt, base_out["persistence"])
                grid[fs][short] = mt

        hz = {
            "k": k, "n_test": int(len(bk)),
            "base_pos_rate": round(base_pos_rate, 4),
            "baselines": base_out,
            "grid": grid,
        }

        # config deploy (best-single chọn trên val ở ĐV5) — kết quả test thật
        if deploy_choice and str(k) in deploy_choice and deploy_choice[str(k)]:
            w = deploy_choice[str(k)]
            short = FULL_TO_SHORT.get(w["model"], w["model"])
            fs = w["feature_set"]
            dm = grid.get(fs, {}).get(short)
            if dm is not None:
                hz["deploy"] = {
                    "feature_set": fs, "model": w["model"],
                    "val_mcc": w.get("val_mcc"), "val_balacc": w.get("val_balacc"),
                    "test": dm,
                    "beats_always_pos": bool(dm["balacc"] > always["balacc"] and not dm["degenerate"]),
                }
        res["horizons"][str(k)] = hz

    # summary gọn cho web/đọc nhanh
    res["summary"] = _summary(res)
    return res


def _summary(res: dict) -> dict:
    """Tóm tắt deploy test mỗi k + đếm horizon có tín hiệu thật."""
    rows, n_signal = [], 0
    for k in HORIZONS:
        h = res["horizons"].get(str(k))
        if not h or "deploy" not in h:
            continue
        d = h["deploy"]
        t = d["test"]
        signal = bool(t["balacc"] > 0.5 and t["mcc"] > 0 and not t["degenerate"])
        n_signal += int(signal)
        rows.append({
            "k": k, "feature_set": d["feature_set"], "model": d["model"],
            "test_acc": round(t["acc"], 4), "test_balacc": round(t["balacc"], 4),
            "test_mcc": round(t["mcc"], 4), "test_auc": t["auc"],
            "pred_pos_rate": round(t["pred_pos_rate"], 3),
            "base_pos_rate": h["base_pos_rate"],
            "delta_balacc_vs_always": d["test"]["delta_vs_always_pos"]["balacc"],
            "degenerate": t["degenerate"], "beats_always_pos": d["beats_always_pos"],
            "has_signal": signal,
        })
    return {"deploy_per_k": rows, "n_horizons_with_signal": n_signal,
            "label": f"{n_signal}/{len(rows)} horizon có tín hiệu test (balacc>0.5 & MCC>0 & không degen)"}


def validate(res: dict) -> None:
    """Hợp đồng results.json; fail → raise (runner bắt, exit 1)."""
    if not res["horizons"]:
        raise ValueError("[evaluate] không có horizon nào được chấm")
    for ks, h in res["horizons"].items():
        for key in ("baselines", "grid", "base_pos_rate", "n_test"):
            if key not in h:
                raise ValueError(f"[evaluate] k={ks} thiếu '{key}'")
        for fs, models in h["grid"].items():
            for short, mt in models.items():
                if not (0.0 <= mt["acc"] <= 1.0 and 0.0 <= mt["balacc"] <= 1.0):
                    raise ValueError(f"[evaluate] k={ks} {fs}/{short}: acc/balacc ngoài [0,1]")
                if not (-1.0 <= mt["mcc"] <= 1.0):
                    raise ValueError(f"[evaluate] k={ks} {fs}/{short}: MCC ngoài [-1,1]")
                c = mt["confusion"]
                if c["tp"] + c["tn"] + c["fp"] + c["fn"] != mt["n"]:
                    raise ValueError(f"[evaluate] k={ks} {fs}/{short}: confusion không khớp n")