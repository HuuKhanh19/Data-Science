"""Step 11 (ĐV5) — Train + Infer (TCB direction, Khung A).

THAY bản walk-forward cũ. Quy trình một-split 80:10:10 (`src/model/split.py`):

    train -> fit (z-score fit CHỈ trên train; chốt leakage 4/4)
    val   -> infer (dùng để CHỌN deploy: val-MCC + chặn degeneracy)
    test  -> infer (CHẠM MỘT LẦN; con số phán quyết để ĐV6 đánh giá)

Với MỖI (feature_set ∈ {l1,eda,full}) × (model ∈ {EN,LGB,LSTM}) × (k ∈ {1,5,10,20}):
fit trên train rồi sinh P(y=+1) cho TOÀN BỘ val + test (kể cả đuôi test nhãn NaN —
giữ cho inference live, ĐV6 loại khi chấm). Hyperparam đã KHÓA ở Step 10 (KHÔNG tune lại).

Output (long, proba-only):
    date | k | feature_set | segment | y_true | en_proba | lgb_proba | lstm_proba
  - segment ∈ {val,test}; căn ĐÚNG (date,k,segment) với predictions_baseline.parquet.
  - mỗi *_proba = P(y=+1) ∈ [0,1]; sign suy ở ĐV6 bằng ngưỡng 0.5 (≥0.5 → +1).
  - model không chạy → cột proba = NaN (chế độ smoke / bỏ model).

Hàm phụ:
  - learning_curves(): data sơ đồ học (LGB logloss/cây, LSTM BCE/epoch, EN val-ll/λ).
  - select_deploy(): per-k best single = argmax val-MCC, loại pred_pos_rate≥0.97
    (degenerate) trừ khi TẤT CẢ degenerate; tiebreak val-balacc.
  - freeze_deploy(): refit config thắng trên TOÀN data tĩnh (rows có nhãn) → serialize.

LSTM (Windows): runner nạp `import torch` TRƯỚC numpy/sklearn/lightgbm (xung đột DLL).
Kiến trúc frozen §5.4: seq_len=20, hidden=32, 1 lớp, head-dropout 0.2, BCE, Adam.
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from src.model.split import (
    HORIZONS, drop_label_nan, feature_columns, make_split, scale_train_test,
)
from src.model.tuning import EN_ALPHA, EN_LAMBDA_GRID, EN_MAX_ITER, SEED

MODELS = ("elastic_net", "lightgbm", "lstm")
PROBA_COLS = ("en_proba", "lgb_proba", "lstm_proba")
COL_OF = {"elastic_net": "en_proba", "lightgbm": "lgb_proba", "lstm": "lstm_proba"}
FEATURE_SET_ORDER = ("l1", "eda", "full")
SEGMENTS = ("val", "test")
CANONICAL = ["date", "k", "feature_set", "segment", "y_true", *PROBA_COLS]

DEGENERATE_POS = 0.97   # pred_pos_rate ≥ ngưỡng này coi như thoái hoá (loại khỏi winner)


@contextlib.contextmanager
def _quiet():
    """Nuốt warning benign lặp mỗi fit: ConvergenceWarning, FutureWarning
    (penalty='elasticnet' sklearn≥1.8 — model vẫn đúng), feature-names của LGBM."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        yield


# ════════════════════════════ tabular: chuẩn bị tập ════════════════════════════
def _tabular_arrays(df, sp, label_col, feats):
    """train/val/test cho EN+LGB: z-score fit-train, nhãn {-1,+1}->{0,1}.

    train: drop NaN nhãn (Khung A train không có NaN nên no-op) → tập FIT.
    val/test: KHÔNG drop — sinh proba cho mọi dòng (đuôi test NaN vẫn dự đoán cho live).
    Trả Xtr_z, ytr01, Xval_z, Xte_z, y_val, y_test, scaler.
    """
    Xtr = df.iloc[sp.train_idx][feats].to_numpy(float)
    ytr = df.iloc[sp.train_idx][label_col].to_numpy(float)
    Xtr, ytr = drop_label_nan(Xtr, ytr)
    Xval = df.iloc[sp.val_idx][feats].to_numpy(float)
    Xte = df.iloc[sp.test_idx][feats].to_numpy(float)
    Xtr_z, Xval_z, Xte_z, scaler = scale_train_test(Xtr, Xval, Xte)
    y_val = df.iloc[sp.val_idx][label_col].to_numpy(float)
    y_te = df.iloc[sp.test_idx][label_col].to_numpy(float)
    return Xtr_z, (ytr > 0).astype(int), Xval_z, Xte_z, y_val, y_te, scaler


# ──────────────────────────── Elastic Net ────────────────────────────
def _en_clf(n_fit: int, lam: float, l1_ratio: float, max_iter: int) -> LogisticRegression:
    """C = 1/(N·λ) theo N của TẬP FIT (λ bất biến, C phụ thuộc N → refit-all khác n_train)."""
    return LogisticRegression(penalty="elasticnet", solver="saga",
                              l1_ratio=l1_ratio, C=1.0 / (n_fit * lam), max_iter=max_iter)


def _fit_predict_en(Xtr_z, ytr01, en_block: dict, *apply_z) -> list[np.ndarray]:
    """Fit EN trên train, trả [proba_per_apply...]. λ khóa ở Step 10."""
    if len(np.unique(ytr01)) < 2:
        return [np.full(len(x), float(ytr01[0])) for x in apply_z]
    clf = _en_clf(len(ytr01), float(en_block["lambda"]),
                  float(en_block.get("l1_ratio", EN_ALPHA)),
                  int(en_block.get("max_iter", EN_MAX_ITER)))
    with _quiet():
        clf.fit(Xtr_z, ytr01)
        pos = list(clf.classes_).index(1)
        return [clf.predict_proba(x)[:, pos] for x in apply_z]


# ──────────────────────────── LightGBM ────────────────────────────
def _lgb_clf(p: dict, seed: int = SEED) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", boosting_type="gbdt",
        learning_rate=float(p.get("learning_rate", 0.05)), bagging_freq=1,
        random_state=seed, n_jobs=-1, verbose=-1,
        num_leaves=int(p["num_leaves"]), min_data_in_leaf=int(p["min_data_in_leaf"]),
        feature_fraction=float(p["feature_fraction"]), bagging_fraction=float(p["bagging_fraction"]),
        lambda_l1=float(p["lambda_l1"]), lambda_l2=float(p["lambda_l2"]),
        n_estimators=int(p["n_estimators"]),
    )


def _fit_predict_lgb(Xtr_z, ytr01, lgb_block: dict, *apply_z,
                     curve_val=None) -> tuple[list[np.ndarray], dict | None]:
    """Fit LGB (n_est đã khóa, KHÔNG early-stop). Nếu curve_val=(Xv,yv) → ghi
    logloss-vs-cây cho sơ đồ học (eval_set chỉ để log, không dừng sớm)."""
    if len(np.unique(ytr01)) < 2:
        return [np.full(len(x), float(ytr01[0])) for x in apply_z], None
    clf = _lgb_clf(lgb_block)
    curve = None
    with _quiet():
        if curve_val is not None:
            Xv, yv = curve_val
            clf.fit(Xtr_z, ytr01, eval_set=[(Xtr_z, ytr01), (Xv, yv)],
                    eval_names=["train", "val"], eval_metric="binary_logloss")
            ev = clf.evals_result_
            curve = {
                "iters": list(range(1, len(ev["train"]["binary_logloss"]) + 1)),
                "train_logloss": [float(v) for v in ev["train"]["binary_logloss"]],
                "val_logloss": [float(v) for v in ev["val"]["binary_logloss"]],
            }
        else:
            clf.fit(Xtr_z, ytr01)
        pos = list(clf.classes_).index(1)
        return [clf.predict_proba(x)[:, pos] for x in apply_z], curve


# ──────────────────────────── LSTM ────────────────────────────
def _torch():
    import torch
    return torch


def _assert_cuda_usable(gpu: int):
    """Fail-fast: bắt cả ca is_available()=True nhưng thiếu kernel (RTX 50xx)."""
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "LSTM cần CUDA nhưng torch.cuda.is_available()=False. "
            "Cài torch cu128 cho RTX 5070 Ti rồi chạy lại (hoặc --models bỏ lstm / --cpu)."
        )
    n = torch.cuda.device_count()
    if not (0 <= gpu < n):
        raise RuntimeError(f"GPU index {gpu} không hợp lệ — thấy {n} card (0..{n-1}).")
    return torch.device(f"cuda:{gpu}")


def _build_lstm_seq(df, sp, label_col, feats, T: int):
    """Sinh chuỗi (·,T,F): scaler fit-train, transform span liên tục a0..a1.

    - train: vị trí trong train_idx đủ lookback T, nhãn quan sát được → tập FIT.
    - val/test: lookback chạm sang đuôi train/embargo = CONTEXT hợp lệ (feature đã
      quan sát; embargo chỉ chặn NHÃN). Sinh chuỗi cho MỌI dòng val/test.
    Trả Xtr,ytr01, Xval, Xte, scaler.
    """
    tr, va, te = sp.train_idx, sp.val_idx, sp.test_idx
    scaler = StandardScaler().fit(df.iloc[tr][feats].to_numpy(float))
    a0 = max(0, min(int(tr[0]), int(va[0]) - T + 1, int(te[0]) - T + 1))
    a1 = int(te[-1])
    Z = scaler.transform(df.iloc[a0:a1 + 1][feats].to_numpy(float))
    y_all = df[label_col].to_numpy(float)

    def seq(pos: int) -> np.ndarray:
        i = pos - a0
        return Z[i - T + 1: i + 1]

    tr_pos = [int(p) for p in tr if (p - a0) - T + 1 >= 0 and not np.isnan(y_all[p])]
    Xtr = np.stack([seq(p) for p in tr_pos]).astype(np.float32)
    ytr = (y_all[tr_pos] > 0).astype(np.float32)
    Xval = np.stack([seq(int(p)) for p in va]).astype(np.float32)
    Xte = np.stack([seq(int(p)) for p in te]).astype(np.float32)
    return Xtr, ytr, Xval, Xte, scaler


def _lstm_module(torch, n_features: int, cfg: dict):
    nn = torch.nn

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, cfg["hidden_size"], cfg["num_layers"],
                                batch_first=True, dropout=cfg.get("lstm_dropout", 0.0))
            self.drop = nn.Dropout(cfg.get("head_dropout", 0.2))
            self.fc1 = nn.Linear(cfg["hidden_size"], cfg["fc_hidden"])
            self.fc2 = nn.Linear(cfg["fc_hidden"], 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            z = self.drop(out[:, -1, :])
            z = torch.relu(self.fc1(z))
            return self.fc2(z).squeeze(-1)        # logits

    return LSTMNet()


def _fit_lstm(Xtr, ytr, cfg: dict, device, want_curve: bool = False):
    """Train LSTM với inner-val (chronological tail) + early stopping. Trả
    (model, scaler-agnostic predict_fn, curve|None). predict_fn(X)->P(y=+1)."""
    torch = _torch()
    seed = int(cfg.get("seed", SEED))
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(Xtr)
    n_inner_val = max(1, int(round(cfg.get("inner_val_frac", 0.15) * n)))
    cut = n - n_inner_val
    Xtr_in, ytr_in = Xtr[:cut], ytr[:cut]
    Xva_in, yva_in = Xtr[cut:], ytr[cut:]

    model = _lstm_module(torch, Xtr.shape[2], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    bs = int(cfg.get("batch_size", 32))

    Xtr_in_t = torch.tensor(Xtr_in, device=device)
    ytr_in_t = torch.tensor(ytr_in, device=device)
    Xva_in_t = torch.tensor(Xva_in, device=device)
    yva_in_t = torch.tensor(yva_in, device=device)

    best = {"val": float("inf"), "state": None, "epoch": -1}
    patience = int(cfg.get("early_stop_patience", 10))
    bad = 0
    curve = {"epochs": [], "train_bce": [], "val_bce": []} if want_curve else None

    for epoch in range(1, int(cfg.get("max_epochs", 100)) + 1):
        model.train()
        perm = torch.randperm(len(Xtr_in_t), device=device)
        run_loss, seen = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logit = model(Xtr_in_t[idx])
            loss = loss_fn(logit, ytr_in_t[idx])
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            seen += len(idx)
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva_in_t), yva_in_t).item() if len(Xva_in_t) else float("nan")
        if curve is not None:
            curve["epochs"].append(epoch)
            curve["train_bce"].append(run_loss / max(seen, 1))
            curve["val_bce"].append(vloss)
        if vloss < best["val"] - 1e-5:
            best = {"val": vloss, "state": {k: v.detach().cpu().clone()
                                            for k, v in model.state_dict().items()},
                    "epoch": epoch}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    def predict_fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logit = model(torch.tensor(X.astype(np.float32), device=device))
            return torch.sigmoid(logit).cpu().numpy()

    if curve is not None:
        curve["best_epoch"] = best["epoch"]
    return model, predict_fn, curve


# ════════════════════════════ orchestration: val+test inference ════════════════════════════
def run(df: pd.DataFrame, hparams: dict, feature_sets: dict,
        models=MODELS, horizons=HORIZONS, feature_set_names=FEATURE_SET_ORDER,
        gpu: int = 0, want_curves: bool = True, verbose: bool = True):
    """Fit train → infer val+test cho mọi (feature_set × model × k).

    Trả (predictions_df [CANONICAL], curves_dict). curves[fs][str(k)][model] = data vẽ.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    do_en, do_lgb, do_lstm = (m in models for m in MODELS)
    device = _assert_cuda_usable(gpu) if do_lstm else None
    lstm_cfg = hparams["lstm"]["config"] if do_lstm else None

    parts, curves = [], {}
    for fs in feature_set_names:
        feats = list(feature_sets[fs])
        curves[fs] = {}
        for k in horizons:
            sp = make_split(df["date"], k)
            label_col = f"y_{k}"
            per_k = hparams["feature_sets"][fs]["per_k"][str(k)]
            ck = {}

            n_val, n_te = len(sp.val_idx), len(sp.test_idx)
            en_v = en_t = lgb_v = lgb_t = lstm_v = lstm_t = None

            if do_en or do_lgb:
                Xtr_z, ytr01, Xval_z, Xte_z, _, _, _ = _tabular_arrays(df, sp, label_col, feats)
                if do_en:
                    en_v, en_t = _fit_predict_en(Xtr_z, ytr01, per_k["elastic_net"], Xval_z, Xte_z)
                    if want_curves:
                        ck["en"] = _en_lambda_curve(Xtr_z, ytr01, Xval_z,
                                                    df.iloc[sp.val_idx][label_col].to_numpy(float),
                                                    float(per_k["elastic_net"]["lambda"]))
                if do_lgb:
                    cv = (Xval_z, (df.iloc[sp.val_idx][label_col].to_numpy(float) > 0).astype(int)) \
                        if want_curves else None
                    (lgb_v, lgb_t), lgbc = _fit_predict_lgb(Xtr_z, ytr01, per_k["lightgbm"],
                                                            Xval_z, Xte_z, curve_val=cv)
                    if lgbc:
                        ck["lgb"] = lgbc
            if do_lstm:
                Xs_tr, ys_tr, Xs_val, Xs_te, _ = _build_lstm_seq(df, sp, label_col, feats,
                                                                 int(lstm_cfg["seq_len"]))
                _, pred_fn, lstmc = _fit_lstm(Xs_tr, ys_tr, lstm_cfg, device, want_curve=want_curves)
                lstm_v, lstm_t = pred_fn(Xs_val), pred_fn(Xs_te)
                if lstmc:
                    ck["lstm"] = lstmc

            for seg, idx, ev, gv, lv in (("val", sp.val_idx, en_v, lgb_v, lstm_v),
                                         ("test", sp.test_idx, en_t, lgb_t, lstm_t)):
                npart = len(idx)
                parts.append(pd.DataFrame({
                    "date": df.iloc[idx]["date"].to_numpy(),
                    "k": k, "feature_set": fs, "segment": seg,
                    "y_true": df.iloc[idx][label_col].to_numpy(float),
                    "en_proba": ev if ev is not None else np.full(npart, np.nan),
                    "lgb_proba": gv if gv is not None else np.full(npart, np.nan),
                    "lstm_proba": lv if lv is not None else np.full(npart, np.nan),
                }))
            curves[fs][str(k)] = ck
            if verbose:
                print(f"[train_infer] fs={fs:<4s} k={k:<2d} val={n_val} test={n_te} "
                      f"| {'EN' if do_en else '··'} {'LGB' if do_lgb else '···'} "
                      f"{'LSTM' if do_lstm else '····'}", flush=True)

    out = pd.concat(parts, ignore_index=True)
    out["k"] = out["k"].astype("int64")
    return out[CANONICAL], curves


def _en_lambda_curve(Xtr_z, ytr01, Xval_z, y_val, chosen_lambda):
    """val-logloss vs λ (re-sweep grid Step 10) cho sơ đồ học EN."""
    yv01 = (y_val > 0).astype(int)
    lls = []
    with _quiet():
        for lam in EN_LAMBDA_GRID:
            clf = _en_clf(len(ytr01), float(lam), EN_ALPHA, EN_MAX_ITER)
            clf.fit(Xtr_z, ytr01)
            p = clf.predict_proba(Xval_z)[:, list(clf.classes_).index(1)]
            lls.append(float(log_loss(yv01, p, labels=[0, 1])))
    return {"lambdas": [float(x) for x in EN_LAMBDA_GRID],
            "val_logloss": lls, "chosen_lambda": float(chosen_lambda)}


# ════════════════════════════ selection: deploy best-single per-k ════════════════════════════
def _metrics_at_half(y_true, proba):
    """MCC/balacc/pred_pos_rate tại ngưỡng 0.5, bỏ dòng y_true NaN."""
    m = ~np.isnan(y_true) & ~np.isnan(proba)
    y = (y_true[m] > 0).astype(int)
    pred = (proba[m] >= 0.5).astype(int)
    two = len(np.unique(y)) > 1 and len(np.unique(pred)) > 1
    return {
        "mcc": float(matthews_corrcoef(y, pred)) if two else 0.0,
        "balacc": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else 0.5,
        "pred_pos_rate": float(pred.mean()) if len(pred) else float("nan"),
        "n": int(m.sum()),
    }


def select_deploy(pred_df: pd.DataFrame) -> dict:
    """Per-k best single trên VAL: argmax val-MCC, LOẠI pred_pos_rate≥0.97 trừ khi
    TẤT CẢ degenerate; tiebreak val-balacc. Trả {str(k): {...winner...}, "rule": ...}."""
    val = pred_df[pred_df["segment"] == "val"]
    choice = {"rule": f"argmax val-MCC; loại pred_pos_rate>={DEGENERATE_POS} trừ khi "
                      f"tất cả degenerate; tiebreak val-balacc"}
    for k in sorted(val["k"].unique()):
        sub = val[val["k"] == k]
        cands = []
        for fs in sub["feature_set"].unique():
            row = sub[sub["feature_set"] == fs]
            for model, col in COL_OF.items():
                proba = row[col].to_numpy(float)
                if np.all(np.isnan(proba)):
                    continue
                met = _metrics_at_half(row["y_true"].to_numpy(float), proba)
                cands.append({"feature_set": str(fs), "model": model, **met,
                              "degenerate": met["pred_pos_rate"] >= DEGENERATE_POS})
        if not cands:
            choice[str(int(k))] = None
            continue
        pool = [c for c in cands if not c["degenerate"]] or cands  # all-degenerate → giữ cả
        pool.sort(key=lambda c: (c["mcc"], c["balacc"]), reverse=True)
        winner = pool[0]
        choice[str(int(k))] = {
            "feature_set": winner["feature_set"], "model": winner["model"],
            "val_mcc": round(winner["mcc"], 4), "val_balacc": round(winner["balacc"], 4),
            "val_pred_pos_rate": round(winner["pred_pos_rate"], 3),
            "all_degenerate": all(c["degenerate"] for c in cands),
            "candidates": sorted(
                [{"feature_set": c["feature_set"], "model": c["model"],
                  "val_mcc": round(c["mcc"], 4), "val_balacc": round(c["balacc"], 4),
                  "val_pred_pos_rate": round(c["pred_pos_rate"], 3),
                  "degenerate": c["degenerate"]} for c in cands],
                key=lambda c: c["val_mcc"], reverse=True),
        }
    return choice


# ════════════════════════════ freeze: refit-on-all ════════════════════════════
def _all_label_rows(df, label_col):
    """Index mọi dòng có nhãn quan sát (toàn series trừ đuôi k NaN)."""
    y = df[label_col].to_numpy(float)
    return np.where(~np.isnan(y))[0]


def freeze_deploy(df: pd.DataFrame, hparams: dict, feature_sets: dict,
                  choice: dict, out_dir, gpu: int = 0, verbose: bool = True) -> dict:
    """Refit config thắng từng k trên TOÀN data tĩnh (rows có nhãn) → serialize.

    EN/LGB: joblib (scaler+clf+meta). LSTM: torch state_dict + scaler (joblib).
    Trả manifest dict {str(k): {...artifact...}}. Cần joblib; LSTM cần torch+CUDA.
    """
    import json
    from pathlib import Path
    import joblib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    manifest = {"note": "refit-on-all (toàn data tĩnh, rows có nhãn) — deploy đóng băng"}
    device = None

    for ks, win in choice.items():
        if ks in ("rule", "note") or win is None:
            continue
        k = int(ks)
        fs, model = win["feature_set"], win["model"]
        feats = list(feature_sets[fs])
        label_col = f"y_{k}"
        per_k = hparams["feature_sets"][fs]["per_k"][str(k)]
        idx = _all_label_rows(df, label_col)
        entry = {"feature_set": fs, "model": model, "features": feats,
                 "val_mcc": win["val_mcc"], "val_balacc": win["val_balacc"],
                 "n_refit": int(len(idx))}

        if model in ("elastic_net", "lightgbm"):
            X = df.iloc[idx][feats].to_numpy(float)
            y01 = (df.iloc[idx][label_col].to_numpy(float) > 0).astype(int)
            scaler = StandardScaler().fit(X)
            Xz = scaler.transform(X)
            with _quiet():
                if model == "elastic_net":
                    clf = _en_clf(len(y01), float(per_k["elastic_net"]["lambda"]),
                                  float(per_k["elastic_net"].get("l1_ratio", EN_ALPHA)),
                                  int(per_k["elastic_net"].get("max_iter", EN_MAX_ITER)))
                else:
                    clf = _lgb_clf(per_k["lightgbm"])
                clf.fit(Xz, y01)
            path = out_dir / f"k{k}_{COL_OF[model].split('_')[0]}.joblib"
            joblib.dump({"scaler": scaler, "clf": clf, "features": feats,
                         "model": model, "k": k}, path)
            entry["artifact"] = path.name

        elif model == "lstm":
            torch = _torch()
            if device is None:
                device = _assert_cuda_usable(gpu)
            cfg = hparams["lstm"]["config"]
            T = int(cfg["seq_len"])
            scaler = StandardScaler().fit(df.iloc[idx][feats].to_numpy(float))
            a0 = max(0, int(idx[0]))
            a1 = int(idx[-1])
            Z = scaler.transform(df.iloc[a0:a1 + 1][feats].to_numpy(float))
            y_all = df[label_col].to_numpy(float)
            pos = [int(p) for p in idx if (p - a0) - T + 1 >= 0]
            X = np.stack([Z[(p - a0) - T + 1:(p - a0) + 1] for p in pos]).astype(np.float32)
            yv = (y_all[pos] > 0).astype(np.float32)
            mdl, _, _ = _fit_lstm(X, yv, cfg, device, want_curve=False)
            path = out_dir / f"k{k}_lstm.pt"
            torch.save({"state_dict": {kk: v.cpu() for kk, v in mdl.state_dict().items()},
                        "cfg": cfg, "n_features": X.shape[2]}, path)
            joblib.dump({"scaler": scaler, "features": feats, "k": k},
                        out_dir / f"k{k}_lstm_scaler.joblib")
            entry["artifact"] = path.name
            entry["scaler_artifact"] = f"k{k}_lstm_scaler.joblib"

        manifest[ks] = entry
        if verbose:
            print(f"[freeze] k={k:<2d} {model:<11s} fs={fs:<4s} refit_n={entry['n_refit']} "
                  f"→ {entry.get('artifact')}", flush=True)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ════════════════════════════ validate ════════════════════════════
def validate(out: pd.DataFrame, baseline: pd.DataFrame | None = None,
             feature_set_names=FEATURE_SET_ORDER) -> None:
    """Hợp đồng output (chế độ FULL). Fail → raise (runner bắt, exit 1)."""
    if list(out.columns) != CANONICAL:
        raise ValueError(f"[train_infer] cột lệch CANONICAL: {list(out.columns)}")
    if set(out["k"].unique()) != set(HORIZONS):
        raise ValueError(f"[train_infer] thiếu horizon: {sorted(out['k'].unique())}")
    if set(out["feature_set"].unique()) != set(feature_set_names):
        raise ValueError(f"[train_infer] thiếu feature_set: {sorted(out['feature_set'].unique())}")
    if set(out["segment"].unique()) != set(SEGMENTS):
        raise ValueError(f"[train_infer] segment phải = {SEGMENTS}")
    for c in PROBA_COLS:
        v = out[c].to_numpy(float)
        v = v[~np.isnan(v)]
        if v.size and (v.min() < 0 or v.max() > 1):
            raise ValueError(f"[train_infer] {c} ngoài [0,1]")
    # mỗi (k,fs,segment) duy nhất số dòng; val không được có NaN y_true (chỉ đuôi test mới NaN)
    val = out[out["segment"] == "val"]
    if val["y_true"].isna().any():
        raise ValueError("[train_infer] val có y_true NaN (sai embargo?)")
    if baseline is not None:
        key = ["date", "k", "segment"]
        m = out[key].drop_duplicates().merge(
            baseline[key].drop_duplicates(), on=key, how="outer", indicator=True)
        if (m["_merge"] != "both").any():
            n_bad = int((m["_merge"] != "both").sum())
            raise ValueError(f"[train_infer] (date,k,segment) lệch baseline: {n_bad} dòng")