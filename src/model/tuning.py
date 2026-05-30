"""Step 10 — Hyperparameter tuning (Phase-0).

Chọn hyperparameter MỘT LẦN trên Phase-0 window ``[0, 1000)``, khóa vào
``config/hparams.json`` — không chạm test (chống p-hacking, anti-pattern #7
research_design §12). Đây là lần tune DUY NHẤT trong cả vòng đời project:
Step 11 và Phase 2 chỉ đọc file đã khóa, không tune lại.

Sơ đồ chọn: **single time-ordered holdout** (KHÔNG k-fold) — val = 15% cuối
Phase-0, embargo gap ``k`` ở ranh giới (đóng leakage overlapping-label một lần)::

    [ ---- train [0, 850-k) ---- | gap k | -- val [850, 1000) -- ]

Tiêu chí: min validation ``binary_logloss``, tune ĐỘC LẬP từng horizon.

    Elastic Net : tune λ (α=0.5 cố định), grid logspace(-4,1,50)
    LightGBM    : tune 6 hyperparam qua grid; n_estimators qua early stopping
    LSTM        : KHÔNG tune — ghi config frozen từ research_design §5.4

Z-score fit-trên-train (tái dùng ``walk_forward.scale_train_test``). Nhãn
``{-1,+1}`` → ``{0,1}`` cho mọi classifier + log-loss. Bước này CPU-only.
"""
from __future__ import annotations

import itertools
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.model.walk_forward import (
    N_TRAIN,
    drop_label_nan,
    feature_columns,
    scale_train_test,
)

HORIZONS = (1, 5, 10, 20)
N_PHASE0 = N_TRAIN          # Phase-0 = [0, 1000)
VAL_FRAC = 0.15             # lát validation = 15% cuối Phase-0
SEED = 42

# --- Elastic Net (research_design §5.2) ---
EN_ALPHA = 0.5                              # tỷ lệ trộn L1/L2 (= l1_ratio sklearn)
EN_LAMBDA_GRID = np.logspace(-4, 1, 50)    # 50 điểm 1e-4 → 10
EN_MAX_ITER = 5000

# --- LightGBM (research_design §5.3) ---
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

# --- LSTM (research_design §5.4) — frozen, KHÔNG tune ---
LSTM_CONFIG = {
    "seq_len": 20,
    "hidden_size": 32,
    "num_layers": 1,
    "lstm_dropout": 0.0,
    "head_dropout": 0.2,
    "fc_hidden": 16,
    "loss": "bce",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 100,
    "early_stop_patience": 10,
    "inner_val_frac": 0.15,
    "seed": SEED,
}


def phase0_split(
    df: pd.DataFrame, k: int, val_frac: float = VAL_FRAC, n_phase0: int = N_PHASE0
) -> tuple[np.ndarray, np.ndarray]:
    """Single holdout trong Phase-0 ``[0, n_phase0)`` với embargo gap ``k``.

    val = ``[n_phase0 - val_size, n_phase0)``; train = ``[0, n_phase0 - val_size - k)``;
    gap ``k`` phiên ở giữa bị loại (nhãn train cuối nhìn k phiên tới = vùng val).
    Trả index thuần (không động nhãn).
    """
    if len(df) < n_phase0:
        raise ValueError(f"cần ≥ {n_phase0} phiên cho Phase-0, có {len(df)}")
    val_size = round(n_phase0 * val_frac)
    val_start = n_phase0 - val_size
    train_end = val_start - k                       # embargo gap k
    if train_end <= 0:
        raise ValueError(f"train rỗng ở k={k}: val_frac quá lớn")
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(val_start, n_phase0)
    return train_idx, val_idx


def _to01(y: np.ndarray) -> np.ndarray:
    """Nhãn {-1,+1} → {0,1} (sklearn/LightGBM/BCE đều cần {0,1})."""
    return (np.asarray(y, dtype=float) > 0).astype(int)


def _assert_phase0_finite(df: pd.DataFrame, feats: list[str]) -> None:
    """Fail-fast: chặn ±inf trong feature Phase-0 trước khi đụng sklearn (lỗi rõ ràng)."""
    arr = df.iloc[:N_PHASE0][feats].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        bad = [c for c in feats
               if not np.isfinite(df.iloc[:N_PHASE0][c].to_numpy(dtype=float)).all()]
        raise ValueError(f"feature Phase-0 có ±inf/NaN: {bad} — sửa pipeline Step 1→7 trước")


def _prepare(df, train_idx, val_idx, label_col, feature_cols):
    """Chọn cột → loại NaN nhãn → z-score (fit trên train) → encode {0,1}.

    Trả ``(X_tr_z, y_tr01, X_val_z, y_val01)`` đã sẵn sàng cho mọi model.
    """
    x_tr = df.iloc[train_idx][feature_cols].to_numpy(dtype=float)
    y_tr = df.iloc[train_idx][label_col].to_numpy(dtype=float)
    x_val = df.iloc[val_idx][feature_cols].to_numpy(dtype=float)
    y_val = df.iloc[val_idx][label_col].to_numpy(dtype=float)
    x_tr, y_tr = drop_label_nan(x_tr, y_tr)
    x_val, y_val = drop_label_nan(x_val, y_val)
    x_tr_z, x_val_z, _ = scale_train_test(x_tr, x_val)   # fit chỉ trên train
    return x_tr_z, _to01(y_tr), x_val_z, _to01(y_val)


def tune_elastic_net(X_tr, y_tr, X_val, y_val) -> dict:
    """Quét λ trên grid, chọn min val log-loss. ``C = 1/(N·λ)`` bám objective pre-reg."""
    n = len(y_tr)
    best = {"val_logloss": np.inf}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)  # sklearn≥1.8 deprecate penalty=
        for lam in EN_LAMBDA_GRID:
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=EN_ALPHA,
                C=1.0 / (n * lam), max_iter=EN_MAX_ITER, tol=1e-3,
                random_state=SEED,
            ).fit(X_tr, y_tr)
            p = clf.predict_proba(X_val)[:, 1]
            ll = log_loss(y_val, p, labels=[0, 1])
            if ll < best["val_logloss"]:
                best = {"lambda": float(lam), "C": float(1.0 / (n * lam)),
                        "val_logloss": float(ll)}
    best["n_train"] = int(n)
    lo, hi = float(EN_LAMBDA_GRID.min()), float(EN_LAMBDA_GRID.max())
    best["at_grid_edge"] = bool(best["lambda"] in (lo, hi))  # cờ: grid có thể hẹp
    return best


def _lgb_grid_iter():
    keys = list(LGB_GRID)
    for combo in itertools.product(*(LGB_GRID[k] for k in keys)):
        yield dict(zip(keys, combo))


def tune_lightgbm(X_tr, y_tr, X_val, y_val) -> dict:
    """Quét grid 6 hyperparam; mỗi tổ hợp chọn n_estimators qua early stopping (val log-loss)."""
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
    best = {"val_logloss": np.inf}
    for grid in _lgb_grid_iter():
        params = {
            "objective": "binary", "boosting_type": "gbdt", "metric": "binary_logloss",
            "learning_rate": LGB_LR, "bagging_freq": 1, "seed": SEED,
            "verbose": -1, "force_row_wise": True, **grid,
        }
        booster = lgb.train(
            params, dtrain, num_boost_round=LGB_N_EST_CAP, valid_sets=[dval],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                       lgb.log_evaluation(0)],
        )
        ll = float(booster.best_score["valid_0"]["binary_logloss"])
        if ll < best["val_logloss"]:
            best = {**grid, "n_estimators": int(booster.best_iteration),
                    "val_logloss": ll}
    best["n_train"] = int(len(y_tr))
    return best


def tune_all(df: pd.DataFrame) -> dict:
    """Tune cả 3 model trên Phase-0, trả dict hparams sẵn sàng ghi JSON."""
    feats = feature_columns(df)
    _assert_phase0_finite(df, feats)
    dates = pd.to_datetime(df["date"])
    val_size = round(N_PHASE0 * VAL_FRAC)
    hp = {
        "meta": {
            "step": 10,
            "tuned_on": f"phase0 [0, {N_PHASE0})",
            "phase0_start": str(dates.iloc[0].date()),
            "phase0_end": str(dates.iloc[N_PHASE0 - 1].date()),
            "n_phase0": N_PHASE0, "val_frac": VAL_FRAC, "val_size": val_size,
            "scheme": "single time-ordered holdout + embargo gap k (KHONG k-fold)",
            "metric": "val_binary_logloss", "seed": SEED, "n_features": len(feats),
        },
        "elastic_net": {
            "fixed": {"penalty": "elasticnet", "solver": "saga",
                      "l1_ratio": EN_ALPHA, "max_iter": EN_MAX_ITER}, "per_horizon": {}},
        "lightgbm": {
            "fixed": {"objective": "binary", "boosting_type": "gbdt",
                      "learning_rate": LGB_LR, "bagging_freq": 1, "seed": SEED}, "per_horizon": {}},
        "lstm": {"note": "frozen tu research_design §5.4, KHONG tune", "config": LSTM_CONFIG},
    }
    for k in HORIZONS:
        tr, va = phase0_split(df, k)
        X_tr, y_tr, X_val, y_val = _prepare(df, tr, va, f"y_{k}", feats)
        hp["elastic_net"]["per_horizon"][str(k)] = tune_elastic_net(X_tr, y_tr, X_val, y_val)
        hp["lightgbm"]["per_horizon"][str(k)] = tune_lightgbm(X_tr, y_tr, X_val, y_val)
    return hp


def validate(hp: dict, df: pd.DataFrame) -> None:
    """Kiểm tra hợp đồng output; fail → raise (runner bắt, exit 1)."""
    # 1. structure — đủ 3 model, đủ 4 horizon
    for m in ("elastic_net", "lightgbm"):
        got = set(hp[m]["per_horizon"])
        assert got == {str(k) for k in HORIZONS}, f"{m} thiếu horizon: {got}"
    assert hp["lstm"]["config"]["hidden_size"] == 32, "LSTM config sai (phải frozen §5.4)"

    # 2. en_lambda_in_grid — λ chọn nằm trong [grid_min, grid_max]
    lo, hi = float(EN_LAMBDA_GRID.min()), float(EN_LAMBDA_GRID.max())
    for k in HORIZONS:
        lam = hp["elastic_net"]["per_horizon"][str(k)]["lambda"]
        assert lo <= lam <= hi, f"EN k={k}: λ={lam} ngoài grid"

    # 3. lgb_in_bounds — hyperparam ∈ grid, n_estimators ∈ [1, cap]
    for k in HORIZONS:
        p = hp["lightgbm"]["per_horizon"][str(k)]
        for key, allowed in LGB_GRID.items():
            assert p[key] in allowed, f"LGB k={k}: {key}={p[key]} ngoài grid"
        assert 1 <= p["n_estimators"] <= LGB_N_EST_CAP, f"LGB k={k}: n_est={p['n_estimators']} ngoài [1,{LGB_N_EST_CAP}]"

    # 4. split_no_leakage — train kết thúc trước val đúng gap k, val đúng val_size
    val_size = round(N_PHASE0 * VAL_FRAC)
    for k in HORIZONS:
        tr, va = phase0_split(df, k)
        assert len(va) == val_size, f"k={k}: val_size={len(va)} ≠ {val_size}"
        assert va[0] - tr[-1] - 1 == k, f"k={k}: gap={va[0]-tr[-1]-1} ≠ {k}"
        assert va[-1] == N_PHASE0 - 1, f"k={k}: val vượt Phase-0"