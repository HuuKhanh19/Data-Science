"""Phase 2 — Sổ theo dõi dự đoán forward (TCB direction, Khung A).

Model ĐÓNG BĂNG (không refit). Mỗi lần chạy inference, dự đoán forward cho phiên
``t+k`` (tương lai CHƯA quan sát) được GHI LẠI vào một sổ append-only. Khi đủ ``k``
phiên giao dịch trôi qua, phiên ``t+k`` đã có giá thật → CHẤM đúng/sai. Nhờ vậy
track record tự kiểm chứng và dài ra theo thời gian — không cần refit, không đụng
phần "past" OOS tĩnh của Phase 1.

Module LOGIC THUẦN (DataFrame in → DataFrame out, không đụng disk). Runner mỏng
Phase 2 lo đọc/ghi ``data/forward_log.parquet``.

Quy ước nhãn GIỮ ĐÚNG Phase 1: ``y = sign(P_{t+k} − P_t)``, tie (``P_{t+k}==P_t``) → +1.
Giá dùng để chấm là **adjusted close** — đúng nguồn đã gán nhãn ở Step 5.

Khoá nghiệp vụ mỗi dòng: ``(from_date, k)`` — một phiên gốc ``t`` chỉ có một dự đoán
cho mỗi horizon. Chạy lại trong ngày = idempotent (upsert dòng CHƯA chấm; KHÔNG bao
giờ ghi đè dòng ĐÃ chấm để giữ lịch sử trung thực).
"""
from __future__ import annotations

import pandas as pd

# Schema khoá cứng của sổ forward.
FWLOG_COLUMNS = [
    "from_date",      # phiên gốc t (ngày HOSE; phiên mới nhất tại thời điểm chạy)
    "k",              # horizon (1/5/10/20)
    "run_date",       # ngày thực sự chạy inference (audit; cập nhật khi upsert)
    "model",          # config deploy đã dùng (audit)
    "feature_set",
    "proba",          # P(y=+1) từ model deploy refit-all trên row mới nhất
    "pred",           # +1 / -1
    "from_price",     # adj close tại t (chốt lúc dự đoán; NaN -> resolve() tự tra)
    "target_date",    # ngày phiên t+k (NaT khi chưa đủ k phiên)
    "target_price",   # adj close tại t+k (NaN khi chưa resolve)
    "y_true",         # +1 / -1 khi đã resolve, else NaN
    "correct",        # True/False khi đã resolve, else NA
    "resolved_at",    # ngày chấm (NaT khi chưa resolve)
]

_DATE_COLS = ("from_date", "run_date", "target_date", "resolved_at")


def empty_log() -> pd.DataFrame:
    """Sổ rỗng đúng schema (dùng cho lần chạy đầu khi chưa có file)."""
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in FWLOG_COLUMNS})
    return _coerce(df)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Ép kiểu chuẩn để so khớp/ghi parquet ổn định."""
    df = df.reindex(columns=FWLOG_COLUMNS).copy()
    for c in _DATE_COLS:
        df[c] = pd.to_datetime(df[c]).dt.normalize()
    df["k"] = df["k"].astype("Int64")
    df["pred"] = df["pred"].astype("Int64")
    df["y_true"] = df["y_true"].astype("Int64")
    df["correct"] = df["correct"].astype("boolean")
    for c in ("proba", "from_price", "target_price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["model"] = df["model"].astype("string")
    df["feature_set"] = df["feature_set"].astype("string")
    return df


def append_predictions(log: pd.DataFrame, new_preds: pd.DataFrame) -> pd.DataFrame:
    """Upsert các dự đoán forward mới vào sổ theo khoá ``(from_date, k)``.

    Quy tắc:
      - Khoá CHƯA tồn tại  -> thêm dòng mới (chưa resolve).
      - Khoá đã có & CHƯA chấm (``correct`` is NA) -> cập nhật proba/pred/run_date...
        (chạy lại cùng ngày, hoặc sửa từ_price).
      - Khoá đã có & ĐÃ chấm -> GIỮ NGUYÊN (không bao giờ viết lại lịch sử đã chấm).

    ``new_preds`` cần tối thiểu: from_date, k, pred, proba; nên kèm run_date, model,
    feature_set, from_price. Cột thiếu được điền mặc định.
    """
    log = _coerce(log)
    new = _coerce(new_preds)

    resolved_keys = set(
        map(tuple, log.loc[log["correct"].notna(), ["from_date", "k"]].to_numpy())
    )
    # Bỏ khỏi new những khoá đã chấm (bất biến lịch sử).
    mask_keep = ~new[["from_date", "k"]].apply(tuple, axis=1).isin(resolved_keys)
    new = new[mask_keep]
    if new.empty:
        return log.sort_values(["from_date", "k"]).reset_index(drop=True)

    # Bỏ khỏi log những khoá CHƯA chấm sắp bị new ghi đè -> rồi nối new vào.
    new_keys = set(map(tuple, new[["from_date", "k"]].to_numpy()))
    mask_drop = (
        log[["from_date", "k"]].apply(tuple, axis=1).isin(new_keys)
        & log["correct"].isna()
    )
    out = pd.concat([log[~mask_drop], new], ignore_index=True)
    return _coerce(out).sort_values(["from_date", "k"]).reset_index(drop=True)


def resolve(log: pd.DataFrame, price: pd.DataFrame, *,
            date_col: str = "date", price_col: str = "adj_close",
            resolved_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Chấm đúng/sai các dòng chưa resolve khi đã đủ ``k`` phiên HOSE.

    ``price`` = bảng giá HOSE (mỗi dòng một phiên giao dịch), có cột ``date`` và
    ``adj_close``. Lịch giao dịch = đúng tập ngày trong ``price`` (không tự sinh).

    Với dòng gốc tại phiên ``t`` (vị trí ``i`` trong lịch): nếu tồn tại phiên ``i+k``
    thì ``P_{t+k}`` đã quan sát -> ``y_true = sign(P_{t+k} − P_t)``, tie → +1,
    ``correct = (pred == y_true)``. Chưa đủ phiên -> để nguyên (sẽ chấm ở lần sau).
    """
    log = _coerce(log)
    if log.empty:
        return log

    px = price[[date_col, price_col]].copy()
    px[date_col] = pd.to_datetime(px[date_col]).dt.normalize()
    px = px.dropna(subset=[date_col]).drop_duplicates(date_col).sort_values(date_col)
    sessions = px[date_col].to_numpy()                      # lịch HOSE (đã sort)
    pos = {d: i for i, d in enumerate(sessions)}            # date -> chỉ số phiên
    close = dict(zip(px[date_col], px[price_col]))          # date -> adj_close
    n = len(sessions)
    resolved_at = pd.Timestamp.now().normalize() if resolved_at is None else pd.Timestamp(resolved_at).normalize()

    log = log.copy()
    todo = log.index[log["correct"].isna()]
    for idx in todo:
        t = log.at[idx, "from_date"]
        k = int(log.at[idx, "k"])
        i = pos.get(t)
        if i is None or i + k >= n:        # phiên gốc chưa có trong lịch, hoặc chưa đủ k phiên
            continue
        t_k = sessions[i + k]
        p0 = log.at[idx, "from_price"]
        if pd.isna(p0):                    # từ_price thiếu (vd seed từ app_data) -> tra từ lịch giá
            p0 = close.get(t)
            log.at[idx, "from_price"] = p0
        if pd.isna(p0):
            continue
        p_k = close[t_k]
        y_true = 1 if (p_k - p0) >= 0 else -1   # tie -> +1, đồng nhất Phase 1
        log.at[idx, "target_date"] = t_k
        log.at[idx, "target_price"] = p_k
        log.at[idx, "y_true"] = y_true
        log.at[idx, "correct"] = bool(int(log.at[idx, "pred"]) == y_true)
        log.at[idx, "resolved_at"] = resolved_at

    return _coerce(log).sort_values(["from_date", "k"]).reset_index(drop=True)


def summary(log: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp track record theo horizon: số đã chấm / đang chờ / hit-rate."""
    log = _coerce(log)
    rows = []
    for k, g in log.groupby("k"):
        done = g[g["correct"].notna()]
        rows.append({
            "k": int(k),
            "n_pending": int(g["correct"].isna().sum()),
            "n_resolved": int(len(done)),
            "n_correct": int(done["correct"].sum()) if len(done) else 0,
            "hit_rate": round(done["correct"].mean(), 4) if len(done) else None,
        })
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)