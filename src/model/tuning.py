"""Step 10 (tune) — Hyperparameter tuning trên VAL theo từng feature_set (Khung A).

THAY bản cũ (Phase-0 [0,1000) single holdout). Tune MỘT LẦN trên ranh giới
train|val của split 80:10:10 (`src/model/split.py`), cho **mỗi feature_set ∈
{l1, eda, full}** (đọc `config/feature_sets.json`). Khóa kết quả vào
`config/hparams.json`. KHÔNG chạm test.

    train  -> fit
    val    -> tune (objective = binary_logloss, mượt & proper scoring)

Tune ĐỘC LẬP từng horizon:
    Elastic Net : tune λ (α=l1_ratio=0.5 cố định), grid logspace(-4,1,50)
    LightGBM    : tune 6 hyperparam qua grid; n_estimators qua early stopping trên val
    LSTM        : KHÔNG tune — ghi config frozen (fit ở Step 11)

Lưu ý chọn (Step 11): hyperparam chọn ở đây theo val-LOGLOSS (ổn định cho search);
phần CHỌN (feature_set × model) deploy theo **val-MCC / balanced-acc** (KHÔNG accuracy
thô — val là regime tăng mạnh) thực hiện ở cuối Step 11. Nhãn {-1,+1}->{0,1}. CPU-only.
"""
from __future__ import annotations

import contextlib
import itertools
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, log_loss, matthews_corrcoef,
)

from src.model.split import (
    HORIZONS, drop_label_nan, make_split, scale_train_test,
)

SEED = 42
FEATURE_SET_ORDER = ("l1", "eda", "full")

# --- Elastic Net ---
EN_ALPHA = 0.5                              # l1_ratio (trộn L1/L2)
EN_LAMBDA_GRID = np.logspace(-4, 1, 50)     # 50 điểm 1e-4 → 10
EN_MAX_ITER = 5000

# --- LightGBM (grid; n_estimators qua early stopping) ---
LGB_LR = 0.05
LGB_N_EST_CAP = 2000
LGB_EARLY_STOP = 50
LGB_GRID = {
    "num_leaves": [15, 31, 63],
    "min_data_in_leaf": [20, 50, 100],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l1": [0, 1],
    "lambda_l2": [0, 1],
}

# --- LSTM frozen-config (không tune; fit ở Step 11) ---
LSTM_CONFIG = {
    "seq_len": 20, "hidden_size": 32, "num_layers": 1,
    "lstm_dropout": 0.0, "head_dropout": 0.2, "fc_hidden": 16,
    "loss": "bce", "optimizer": "adam", "learning_rate": 1e-3,
    "batch_size": 32, "max_epochs": 100, "early_stop_patience": 10,
    "inner_val_frac": 0.15, "seed": SEED,
}


@contextlib.contextmanager
def _quiet():
    """Nuốt warning benign lặp lại mỗi fit/predict: ConvergenceWarning,
    FutureWarning penalty='elasticnet' (sklearn≥1.8, model vẫn đúng),
    UserWarning feature-names của LGBM khi predict trên numpy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        yield


def _val_metrics(y01: np.ndarray, proba: np.ndarray) -> dict:
    """logloss + acc/balacc/MCC tại ngưỡng 0.5 (để theo dõi; chọn cuối dùng MCC/balacc)."""
    pred = (proba >= 0.5).astype(int)
    two_class = len(np.unique(y01)) > 1 and len(np.unique(pred)) > 1
    return {
        "logloss": float(log_loss(y01, proba, labels=[0, 1])),
        "acc": float(accuracy_score(y01, pred)),
        "balacc": float(balanced_accuracy_score(y01, pred)),
        "mcc": float(matthews_corrcoef(y01, pred)) if two_class else 0.0,
        "pred_pos_rate": float(pred.mean()),
    }


def _prepare(df, idx_tr, idx_val, label_col, feats):
    """Chọn cột feature_set -> lọc NaN nhãn -> z-score fit-train -> {0,1}."""
    Xtr = df.iloc[idx_tr][feats].to_numpy(float)
    ytr = df.iloc[idx_tr][label_col].to_numpy(float)
    Xval = df.iloc[idx_val][feats].to_numpy(float)
    yval = df.iloc[idx_val][label_col].to_numpy(float)
    Xtr, ytr = drop_label_nan(Xtr, ytr)
    Xval, yval = drop_label_nan(Xval, yval)
    Xtr_z, Xval_z, _ = scale_train_test(Xtr, Xval)
    return Xtr_z, (ytr > 0).astype(int), Xval_z, (yval > 0).astype(int)


def tune_elastic_net(X_tr, y_tr, X_val, y_val) -> dict:
    """Grid λ, chọn min val logloss. C = 1/(N·λ) theo N của train."""
    n = len(y_tr)
    best = None
    with _quiet():
        for lam in EN_LAMBDA_GRID:
            clf = LogisticRegression(penalty="elasticnet", solver="saga",
                                     l1_ratio=EN_ALPHA, C=1.0 / (n * lam),
                                     max_iter=EN_MAX_ITER)
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_val)[:, 1]
            ll = log_loss(y_val, p, labels=[0, 1])
            if best is None or ll < best[0]:
                best = (ll, float(lam), p)
    _, lam, p = best
    edge = lam in (float(EN_LAMBDA_GRID.min()), float(EN_LAMBDA_GRID.max()))
    return {
        "lambda": lam, "l1_ratio": EN_ALPHA, "C": 1.0 / (n * lam),
        "max_iter": EN_MAX_ITER, "lambda_at_grid_edge": bool(edge),
        "n_train": int(n), "val": _val_metrics(y_val, p),
    }


def tune_lightgbm(X_tr, y_tr, X_val, y_val, grid: dict | None = None) -> dict:
    """Grid 6 hyperparam; n_estimators qua early stopping; chọn min val logloss."""
    if grid is None:
        grid = LGB_GRID
    keys = list(grid)
    best = None
    with _quiet():
        for combo in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, combo))
            clf = LGBMClassifier(objective="binary", boosting_type="gbdt",
                                 learning_rate=LGB_LR, n_estimators=LGB_N_EST_CAP,
                                 bagging_freq=1, random_state=SEED, verbose=-1, **params)
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False)])
            n_est = clf.best_iteration_ or LGB_N_EST_CAP
            p = clf.predict_proba(X_val, num_iteration=n_est)[:, 1]
            ll = log_loss(y_val, p, labels=[0, 1])
            if best is None or ll < best[0]:
                best = (ll, dict(params), int(n_est), p)
    _, params, n_est, p = best
    return {**params, "n_estimators": n_est, "learning_rate": LGB_LR,
            "n_train": int(len(y_tr)), "val": _val_metrics(y_val, p)}


def tune_all(df: pd.DataFrame, feature_sets: dict, verbose: bool = False) -> dict:
    """Tune EN+LGB cho mỗi (feature_set, horizon). Trả dict hparams đầy đủ.

    verbose=True: in tiến độ từng (feature_set, k) ngay khi xong (theo dõi run dài).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    n_combo = 1
    for v in LGB_GRID.values():
        n_combo *= len(v)
    if verbose:
        print(f"[tune] bắt đầu | {len(FEATURE_SET_ORDER)} feature_set x {len(HORIZONS)} horizon "
              f"| EN grid={len(EN_LAMBDA_GRID)} λ, LGB grid={n_combo} tổ hợp/horizon", flush=True)
    t0 = time.perf_counter()
    hp = {
        "meta": {
            "step": "10-tune",
            "scheme": "80:10:10 time split + embargo k; tune tren VAL; objective=val_logloss",
            "select_rule": "deploy = argmax val-MCC over (feature_set x model) o Step 11 (KHONG accuracy)",
            "seed": SEED,
            "en_lambda_grid": [float(EN_LAMBDA_GRID.min()), float(EN_LAMBDA_GRID.max()), len(EN_LAMBDA_GRID)],
            "lgb_grid": LGB_GRID,
        },
        "lstm": {"frozen": True, "config": LSTM_CONFIG,
                 "note": "khong tune; fit o Step 11 cho moi feature_set"},
        "feature_sets": {},
    }
    for fs in FEATURE_SET_ORDER:
        feats = feature_sets[fs]
        if verbose:
            print(f"[tune] === feature_set='{fs}' ({len(feats)} feature) ===", flush=True)
        per_k = {}
        for k in HORIZONS:
            tk = time.perf_counter()
            sp = make_split(df["date"], k)
            X_tr, y_tr, X_val, y_val = _prepare(df, sp.train_idx, sp.val_idx, f"y_{k}", feats)
            en = tune_elastic_net(X_tr, y_tr, X_val, y_val)
            lg = tune_lightgbm(X_tr, y_tr, X_val, y_val)
            per_k[str(k)] = {"elastic_net": en, "lightgbm": lg}
            if verbose:
                dt = time.perf_counter() - tk
                ev, gv = en["val"], lg["val"]
                print(f"[tune]   k={k:<2d} ({dt:5.1f}s) "
                      f"EN λ={en['lambda']:.4g} mcc={ev['mcc']:+.3f} balacc={ev['balacc']:.3f} "
                      f"pos={ev['pred_pos_rate']:.2f} ll={ev['logloss']:.4f} | "
                      f"LGB n_est={lg['n_estimators']:<4d} mcc={gv['mcc']:+.3f} balacc={gv['balacc']:.3f} "
                      f"pos={gv['pred_pos_rate']:.2f} ll={gv['logloss']:.4f}", flush=True)
        hp["feature_sets"][fs] = {"features": list(feats), "per_k": per_k}
    if verbose:
        print(f"[tune] xong toàn bộ trong {time.perf_counter() - t0:.1f}s", flush=True)
    return hp


def validate(hp: dict, feature_sets: dict) -> None:
    """Hợp đồng output; fail → raise (runner bắt, exit 1)."""
    lo, hi = float(EN_LAMBDA_GRID.min()), float(EN_LAMBDA_GRID.max())
    assert hp["lstm"]["config"]["hidden_size"] == 32, "LSTM config sai (phải frozen)"
    for fs in FEATURE_SET_ORDER:
        assert fs in hp["feature_sets"], f"thiếu feature_set {fs}"
        assert hp["feature_sets"][fs]["features"] == list(feature_sets[fs]), \
            f"{fs}: danh sách feature lệch feature_sets.json"
        per_k = hp["feature_sets"][fs]["per_k"]
        assert set(per_k) == {str(k) for k in HORIZONS}, f"{fs}: thiếu horizon"
        for k in HORIZONS:
            en = per_k[str(k)]["elastic_net"]
            assert lo <= en["lambda"] <= hi, f"{fs} k={k}: λ ngoài grid"
            g = per_k[str(k)]["lightgbm"]
            for key, allowed in LGB_GRID.items():
                assert g[key] in allowed, f"{fs} k={k} LGB: {key}={g[key]} ngoài grid"
            assert 1 <= g["n_estimators"] <= LGB_N_EST_CAP, \
                f"{fs} k={k}: n_est ngoài [1,{LGB_N_EST_CAP}]"