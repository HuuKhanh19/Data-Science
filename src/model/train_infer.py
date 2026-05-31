"""Step 11 — Walk-forward train + inference (TCB direction).

Đọc features.parquet + hparams.json (ĐÃ KHÓA ở Step 10, KHÔNG tune lại), refit hàng
tuần trên đúng lịch walk-forward (Step 8), dự đoán P(y=+1) cho từng phiên test của
3 model độc lập × 4 horizon. Đây là LẦN ĐẦU pipeline chạm test set [1000, 1742).

    elastic_net : Logistic + Elastic Net, C = 1/(N·λ) tính theo N của TỪNG window
    lightgbm    : GBDT, n_estimators đã chốt ở Step 10 (early stopping), không val lại
    lstm        : kiến trúc frozen §5.4, chuỗi T=20, GPU; strict windowing

Output (long format, proba-only): date | k | y_true | en_proba | lgb_proba | lstm_proba
  - mỗi *_proba = P(y=+1) ∈ [0,1]; sign suy ở Step 12 bằng ngưỡng 0.5 (≥0.5 → +1).
  - y_true giữ NaN ở đuôi k phiên (model vẫn dự đoán cho inference live; Step 12 loại).
  - tập (date,k) khớp ĐÚNG predictions_baseline.parquet → DM so cùng điểm.

Quy ước chống leakage tái dùng từ Step 8: buffer gap k + z-score fit-trên-train.
LSTM windowing (chốt 30/05/2026):
  - STRICT: chuỗi train không thò ra trước mốc train_start → bỏ (T-1) mẫu train đầu/window.
  - gap-rows được dùng làm CONTEXT cho chuỗi test (feature đã quan sát; gap chỉ chặn nhãn).
  - z-score: μ,σ fit CHỈ trên 1000 phiên train, apply cho mọi row dùng trong chuỗi.
"""
from __future__ import annotations

import warnings

# Nạp torch TRƯỚC numpy/sklearn/lightgbm. Trên Windows, nạp torch SAU các thư viện
# kéo MKL/OpenMP (numpy, sklearn, lightgbm) làm c10.dll init fail (WinError 1114) do
# xung đột thứ tự DLL. try/except để EN/LGB vẫn chạy được khi torch vắng/hỏng.
try:
    import torch as _torch  # noqa: F401  (preload cố định thứ tự nạp DLL)
except Exception:
    _torch = None

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.model.walk_forward import (
    build_window,
    feature_columns,
    walk_forward_splits,
)

HORIZONS = (1, 5, 10, 20)                       # gương theo src/data/label.py
MODELS = ("elastic_net", "lightgbm", "lstm")
PROBA_COLS = ("en_proba", "lgb_proba", "lstm_proba")
CANONICAL = ["date", "k", "y_true", *PROBA_COLS]


@__import__("contextlib").contextmanager
def _quiet():
    """Nuốt warning benign lặp lại mỗi window: ConvergenceWarning, deprecation
    penalty='elasticnet' (sklearn 1.8, model vẫn đúng), feature-names của LGBM."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        yield


# ──────────────────────────── Elastic Net ────────────────────────────
def _fit_predict_en(X_tr, y01, X_te, en_block: dict, k: int) -> np.ndarray:
    """Logistic + Elastic Net. λ khóa ở Step 10; C = 1/(N·λ) theo N của window này.

    λ là hệ số regularization của objective pre-reg (chuẩn hóa 1/N trên loss), nên
    BẤT BIẾN theo N — chỉ C của sklearn phải tính lại với N = số mẫu train hiện tại.
    """
    if len(np.unique(y01)) < 2:                 # window 1 lớp (cực hiếm) → hằng
        return np.full(len(X_te), float(y01[0]))
    fixed = en_block["fixed"]
    lam = float(en_block["per_horizon"][str(k)]["lambda"])
    C = 1.0 / (len(y01) * lam)
    clf = LogisticRegression(
        penalty=fixed["penalty"], solver=fixed["solver"],
        l1_ratio=fixed["l1_ratio"], max_iter=fixed["max_iter"], C=C,
    )
    with _quiet():
        clf.fit(X_tr, y01)
        return clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]


# ──────────────────────────── LightGBM ────────────────────────────
def _fit_predict_lgb(X_tr, y01, X_te, lgb_block: dict, k: int) -> np.ndarray:
    """GBDT với hyperparam đã khóa; n_estimators cố định (đã early-stop ở Step 10)."""
    if len(np.unique(y01)) < 2:
        return np.full(len(X_te), float(y01[0]))
    fixed = lgb_block["fixed"]
    p = lgb_block["per_horizon"][str(k)]
    clf = LGBMClassifier(
        objective="binary", boosting_type="gbdt",
        learning_rate=fixed["learning_rate"], bagging_freq=fixed["bagging_freq"],
        random_state=fixed["seed"], n_jobs=-1, verbose=-1,
        num_leaves=p["num_leaves"], min_data_in_leaf=p["min_data_in_leaf"],
        feature_fraction=p["feature_fraction"], bagging_fraction=p["bagging_fraction"],
        lambda_l1=p["lambda_l1"], lambda_l2=p["lambda_l2"],
        n_estimators=p["n_estimators"],
    )
    with _quiet():
        clf.fit(X_tr, y01)
        return clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]


# ──────────────────────────── LSTM ────────────────────────────
def _assert_cuda_usable(gpu: int = 0):
    """Fail-fast: bắt cả ca is_available()=True nhưng thiếu kernel sm_120 (RTX 50xx).
    gpu = index card muốn dùng (0-based)."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "LSTM cần CUDA nhưng torch.cuda.is_available()=False. "
            "Cài torch bản cu128 cho RTX 5070 Ti rồi chạy lại."
        )
    n = torch.cuda.device_count()
    if not (0 <= gpu < n):
        raise RuntimeError(f"GPU index {gpu} không hợp lệ — thấy {n} card (0..{n - 1}).")
    torch.cuda.set_device(gpu)                  # mọi alloc ngầm cũng rơi vào card này
    device = torch.device(f"cuda:{gpu}")
    try:
        x = torch.randn(64, 64, device=device)
        _ = (x @ x).sum().item()
    except RuntimeError as e:  # no kernel image (sm_120 thiếu) hoặc lỗi GPU khác
        raise RuntimeError(
            "GPU không chạy được kernel CUDA (rất có thể torch thiếu sm_120 cho "
            "RTX 50xx). Vá torch (cu128) rồi chạy lại. Chi tiết gốc: " + str(e)
        )
    return device


def _set_torch_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # best-effort; residual stochasticity §5.4
    torch.backends.cudnn.benchmark = False


def _build_lstm(input_size: int, cfg: dict):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, cfg["hidden_size"], cfg["num_layers"],
                batch_first=True, dropout=cfg["lstm_dropout"],
            )
            self.head = nn.Sequential(
                nn.Dropout(cfg["head_dropout"]),
                nn.Linear(cfg["hidden_size"], cfg["fc_hidden"]),
                nn.ReLU(),
                nn.Linear(cfg["fc_hidden"], 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)               # (B, T, H)
            return torch.sigmoid(self.head(out[:, -1, :])).squeeze(-1)  # P(y=+1)

    return Net()


def _lstm_window(df, train_idx, test_idx, label_col, feature_cols, T):
    """Dựng chuỗi cho 1 window (strict, gap-as-context, z-score fit-trên-train).

    Trả: (X_tr_seq (n_tr,T,F), y_tr01, X_te_seq (n_te,T,F), test_dates).
    """
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    a0, a1 = int(train_idx[0]), int(test_idx[-1])
    scaler = StandardScaler().fit(df.iloc[train_idx][feature_cols].to_numpy(float))
    Z = scaler.transform(df.iloc[a0:a1 + 1][feature_cols].to_numpy(float))  # row ↔ pos a0..a1
    y_all = df[label_col].to_numpy(float)

    def seq(pos: int) -> np.ndarray:
        i = pos - a0                            # offset trong block
        return Z[i - T + 1: i + 1]              # (T, F)

    # train: STRICT trong block train, đủ T phiên, nhãn quan sát được
    tr_pos = [p for p in range(a0 + T - 1, int(train_idx[-1]) + 1) if not np.isnan(y_all[p])]
    X_tr = np.stack([seq(p) for p in tr_pos]).astype(np.float32)
    y_tr = (y_all[tr_pos] > 0).astype(np.float32)

    te_pos = [int(p) for p in test_idx]         # lookback chạm gap = context hợp lệ
    X_te = np.stack([seq(p) for p in te_pos]).astype(np.float32)
    test_dates = pd.to_datetime(df.iloc[test_idx]["date"]).to_numpy()
    return X_tr, y_tr, X_te, test_dates


def _fit_predict_lstm(X_tr, y_tr, X_te, cfg: dict, device) -> np.ndarray:
    """Train LSTM 1 window với inner-val early stopping; trả P(y=+1) cho test."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _set_torch_seed(cfg["seed"])
    n = len(X_tr)
    n_val = max(1, round(n * cfg["inner_val_frac"]))     # 15% cuối train (time-ordered)
    Xt, yt = X_tr[:-n_val], y_tr[:-n_val]
    Xv, yv = X_tr[-n_val:], y_tr[-n_val:]

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Xt_t, yt_t, Xv_t, yv_t = t(Xt), t(yt), t(Xv), t(yv)
    model = _build_lstm(X_tr.shape[-1], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.BCELoss()
    loader = DataLoader(TensorDataset(Xt_t, yt_t), batch_size=cfg["batch_size"], shuffle=True)

    best, best_state, ctr = float("inf"), None, 0
    for _ in range(cfg["max_epochs"]):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv_t), yv_t).item()
        if vl < best - 1e-5:
            best, ctr = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            ctr += 1
            if ctr >= cfg["early_stop_patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(t(X_te)).cpu().numpy()


# ──────────────────────────── Orchestrator ────────────────────────────
def run(df: pd.DataFrame, hparams: dict, horizons=HORIZONS, models=MODELS, gpu: int = 0) -> pd.DataFrame:
    """Refit walk-forward + inference cho mọi (model, horizon) yêu cầu → long format.

    Model không yêu cầu → cột proba tương ứng = NaN (chế độ smoke). Vòng lặp đi qua
    splitter MỘT LẦN mỗi k nên thứ tự (date) của 3 model luôn khớp nhau. gpu = index
    card cho LSTM (mặc định 0).
    """
    feats = feature_columns(df)
    do_en, do_lgb, do_lstm = (m in models for m in MODELS)
    device = _assert_cuda_usable(gpu) if do_lstm else None
    cfg = hparams["lstm"]["config"] if do_lstm else None

    parts = []
    for k in horizons:
        label_col = f"y_{k}"
        y_full = df[label_col].to_numpy(float)
        for train_idx, test_idx in walk_forward_splits(df["date"], k):
            test_idx = np.asarray(test_idx)
            n = len(test_idx)
            en_p = np.full(n, np.nan)
            lgb_p = np.full(n, np.nan)
            lstm_p = np.full(n, np.nan)

            if do_en or do_lgb:
                X_tr, y_tr, X_te, _, _, _ = build_window(df, train_idx, test_idx, label_col, feats)
                y01 = (y_tr > 0).astype(int)
                if do_en:
                    en_p = _fit_predict_en(X_tr, y01, X_te, hparams["elastic_net"], k)
                if do_lgb:
                    lgb_p = _fit_predict_lgb(X_tr, y01, X_te, hparams["lightgbm"], k)
            if do_lstm:
                Xs_tr, ys_tr, Xs_te, _ = _lstm_window(df, train_idx, test_idx, label_col, feats, cfg["seq_len"])
                lstm_p = _fit_predict_lstm(Xs_tr, ys_tr, Xs_te, cfg, device)

            parts.append(pd.DataFrame({
                "date": pd.to_datetime(df.iloc[test_idx]["date"]).to_numpy(),
                "k": k,
                "y_true": y_full[test_idx],
                "en_proba": en_p, "lgb_proba": lgb_p, "lstm_proba": lstm_p,
            }))

    out = pd.concat(parts, ignore_index=True)
    out["k"] = out["k"].astype("int64")
    return out[CANONICAL]


def validate(out: pd.DataFrame, baseline: pd.DataFrame) -> None:
    """Hợp đồng output (chỉ chạy ở chế độ FULL); fail → raise."""
    # 1. schema
    assert list(out.columns) == CANONICAL, f"schema lệch: {list(out.columns)}"
    assert out["k"].dtype.kind == "i", "k phải integer"
    for c in PROBA_COLS:
        assert out[c].dtype.kind == "f", f"{c} phải float"

    # 2. proba_domain — finite, ∈ [0,1], không NaN (full mode → mọi model đã chạy)
    for c in PROBA_COLS:
        v = out[c].to_numpy()
        assert np.isfinite(v).all(), f"{c} có NaN/inf"
        assert ((v >= 0.0) & (v <= 1.0)).all(), f"{c} ngoài [0,1]"

    # 3. coverage — (date,k) khớp ĐÚNG baseline, không trùng
    assert not out.duplicated(["date", "k"]).any(), "trùng (date,k)"
    _key = lambda d: set(
        (pd.Timestamp(t), int(kk))
        for t, kk in zip(pd.to_datetime(d["date"]), d["k"])
    )
    assert _key(out) == _key(baseline), "coverage (date,k) lệch baseline"

    # 4. y_true_tail_nan — mỗi k đúng k NaN ở đuôi (theo date)
    for k in HORIZONS:
        nan_mask = out.loc[out["k"] == k].sort_values("date")["y_true"].isna().to_numpy()
        assert nan_mask.sum() == k, f"y_true k={k}: {int(nan_mask.sum())} NaN, chờ {k}"
        assert nan_mask[-k:].all() and not nan_mask[:-k].any(), f"NaN k={k} không ở đuôi"