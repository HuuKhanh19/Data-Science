"""ĐV7 — Web inference (TCB direction, Khung A). Logic thuần, không I/O frontend.

Ráp `web/app_data.json` cho web local (Phase 1 tĩnh, khoá 2026-05-29; Phase 2 chạy
lại pipeline → rebuild). INFERENCE-ONLY: KHÔNG refit, KHÔNG tune.

Leakage-aware (Phương án A):
  - PAST predictions (track record quá khứ) = lấy từ `predictions_model.parquet`
    (model fit-train, dự đoán OOS trên val+test) — KHÔNG chạy model refit-all trên
    quá khứ (sẽ in-sample, gian dối).
  - FORWARD prediction (phiên k tới) = model DEPLOY refit-on-all (đóng băng ĐV5) chạy
    trên feature row MỚI NHẤT — đây là chỗ DUY NHẤT model all-data hợp lệ (tương lai
    thật chưa quan sát).
  - Reliability mỗi horizon = đọc `results.json` (test chạm 1 lần ĐV6); web HIỂN THỊ
    trung thực (badge tín hiệu/yếu/không-tin-cậy), KHÔNG để test re-pick deploy.

deploy/ artifact (ĐV5 freeze): k{K}_{lgb,en}.joblib | k{K}_lstm.pt + k{K}_lstm_scaler.joblib
+ manifest.json. EN/LGB self-contained (joblib). LSTM cần torch + class builder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 20)
SHORT_OF = {"elastic_net": "en", "lightgbm": "lgb", "lstm": "lstm"}
PROBA_OF = {"elastic_net": "en_proba", "lightgbm": "lgb_proba", "lstm": "lstm_proba"}


def _read_json(path: Path):
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Không decode được {path}")


# ──────────────────────────── forward inference (deploy refit-all) ────────────────────────────
def _predict_forward(df: pd.DataFrame, k: int, entry: dict, deploy_dir: Path) -> dict:
    """Chạy model deploy đóng băng trên feature row mới nhất → P(y=+1) cho +k phiên."""
    import joblib
    feats = entry["features"]
    model = entry["model"]
    art = deploy_dir / entry["artifact"]
    last_date = pd.to_datetime(df["date"]).iloc[-1]

    if model in ("elastic_net", "lightgbm"):
        import warnings
        b = joblib.load(art)
        x = df.iloc[[-1]][b["features"]].to_numpy(float)
        xz = b["scaler"].transform(x)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            proba = float(b["clf"].predict_proba(xz)[:, list(b["clf"].classes_).index(1)][0])
    elif model == "lstm":
        import torch
        from src.model.train_infer import _lstm_module
        ckpt = torch.load(art, map_location="cpu", weights_only=False)
        sc = joblib.load(deploy_dir / entry["scaler_artifact"])
        T = int(ckpt["cfg"]["seq_len"])
        net = _lstm_module(torch, ckpt["n_features"], ckpt["cfg"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        rows = df.iloc[-T:][sc["features"]].to_numpy(float)
        if len(rows) < T:                       # đệm đầu nếu thiếu (cực hiếm)
            rows = np.vstack([np.repeat(rows[:1], T - len(rows), axis=0), rows])
        seq = sc["scaler"].transform(rows).astype(np.float32)[None, :, :]
        with torch.no_grad():
            proba = float(torch.sigmoid(net(torch.tensor(seq))).item())
    else:
        raise ValueError(f"model lạ: {model}")

    return {
        "from_date": str(last_date.date()),
        "horizon_label": f"+{k} phiên",
        "target_date_est": str((last_date + pd.tseries.offsets.BDay(k)).date()),
        "proba": round(proba, 4),
        "pred": 1 if proba >= 0.5 else -1,
        "direction": "TĂNG" if proba >= 0.5 else "GIẢM",
    }


# ──────────────────────────── reliability / badge ────────────────────────────
def _badge(rel: dict | None) -> dict:
    """Phương án A — nhãn trung thực mỗi horizon từ test ĐV6."""
    if rel is None:
        return {"tier": "unknown", "label": "chưa có đánh giá test", "color": "muted"}
    t = rel["test"]
    if t.get("degenerate"):
        return {"tier": "unreliable", "label": "thoái hoá (luôn +1) — không tin cậy", "color": "red"}
    if not rel.get("beats_always_pos", False) or t["balacc"] <= 0.5 or t["mcc"] <= 0:
        return {"tier": "unreliable", "label": "không vượt baseline — không tin cậy", "color": "red"}
    return {"tier": "weak", "label": "tín hiệu yếu (hơi trên ngẫu nhiên)", "color": "amber"}


# ──────────────────────────── assemble ────────────────────────────
def assemble_app_data(features_pq: Path, price_pq: Path, predictions_pq: Path,
                      results_json: Path, deploy_dir: Path,
                      freeze_date: str = "2026-05-29", phase: int = 1) -> dict:
    """Ráp toàn bộ payload cho frontend."""
    df = pd.read_parquet(features_pq)
    df["date"] = pd.to_datetime(df["date"])
    price = pd.read_parquet(price_pq)[["date", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"])
    preds = pd.read_parquet(predictions_pq)
    preds["date"] = pd.to_datetime(preds["date"])
    results = _read_json(results_json)
    manifest = _read_json(deploy_dir / "manifest.json")

    # giá: full history (adjusted close), nghìn VND
    price = price.dropna(subset=["close"]).sort_values("date")
    price_series = [{"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 3)}
                    for d, c in zip(price["date"], price["close"])]

    summary = results.get("summary", {})
    horizons = {}
    for k in HORIZONS:
        ks = str(k)
        man = manifest.get(ks)
        hz_res = results.get("horizons", {}).get(ks, {})
        deploy_rel = hz_res.get("deploy")  # có test metrics nếu deploy có trong grid

        # past OOS predictions từ predictions_model (deploy config)
        past = []
        val_test_boundary = None
        if man:
            fs, model = man["feature_set"], man["model"]
            col = PROBA_OF[model]
            sub = preds[(preds["k"] == k) & (preds["feature_set"] == fs)].copy()
            sub = sub[sub[col].notna() & sub["y_true"].notna()].sort_values("date")
            for d, seg, proba, yt in zip(sub["date"], sub["segment"], sub[col], sub["y_true"]):
                pred = 1 if proba >= 0.5 else -1
                past.append({
                    "date": d.strftime("%Y-%m-%d"), "segment": seg,
                    "proba": round(float(proba), 4), "pred": pred,
                    "y_true": int(yt), "correct": bool(pred == int(yt)),
                })
            tests = [p["date"] for p in past if p["segment"] == "test"]
            if tests:
                val_test_boundary = min(tests)

        forward = None
        if man:
            try:
                forward = _predict_forward(df, k, man, deploy_dir)
            except Exception as e:  # forward best-effort (vd thiếu torch cho lstm)
                forward = {"error": f"{type(e).__name__}: {e}"}

        rel = None
        if deploy_rel:
            rel = {
                "test": deploy_rel["test"],
                "beats_always_pos": deploy_rel.get("beats_always_pos", False),
                "val_mcc": deploy_rel.get("val_mcc"),
                "base_pos_rate": hz_res.get("base_pos_rate"),
                "n_test": hz_res.get("n_test"),
            }

        horizons[ks] = {
            "k": k,
            "deploy": ({"feature_set": man["feature_set"], "model": man["model"]} if man else None),
            "reliability": rel,
            "badge": _badge(rel),
            "forward": forward,
            "past": past,
            "val_test_boundary": val_test_boundary,
        }

    return {
        "meta": {
            "ticker": "TCB", "exchange": "HOSE",
            "phase": phase, "freeze_date": freeze_date,
            "price_unit": "nghìn VND (adjusted close)",
            "last_price_date": price_series[-1]["date"] if price_series else None,
            "n_horizons_signal": summary.get("n_horizons_with_signal", 0),
            "summary_label": summary.get("label", ""),
            "note_phaseA": ("Deploy chọn trên val (ĐV5), test chạm 1 lần (ĐV6) chỉ để "
                            "BÁO CÁO trung thực — không dùng test để đổi model."),
        },
        "price": price_series,
        "horizons": horizons,
    }


def validate(app: dict) -> None:
    """Hợp đồng app_data; fail → raise."""
    if not app.get("price"):
        raise ValueError("[web] price rỗng")
    if set(app["horizons"].keys()) != {str(k) for k in HORIZONS}:
        raise ValueError(f"[web] thiếu horizon: {sorted(app['horizons'])}")
    for ks, h in app["horizons"].items():
        if h["deploy"] is None:
            raise ValueError(f"[web] k={ks} thiếu deploy (manifest?)")
        if not isinstance(h["past"], list):
            raise ValueError(f"[web] k={ks} past sai kiểu")