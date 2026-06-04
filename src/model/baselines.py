"""Step 9 — Baselines (TCB direction, Khung A).

Sinh dự đoán 3 baseline tầm thường trên ĐÚNG các phiên VAL + TEST của split mới
(`src/model/split.py`), làm RÀO tham chiếu cho Step 12 (so với model). Không học,
không GPU — thuần quy tắc. THAY bản cũ chấm theo tuần walk-forward.

    persistence  : ŷ_{t,k} = sign(P_t − P_{t-k}) = y_{·,k}.shift(k)
                   (= nhãn đã hiện thực hóa k phiên trước; không cần giá thô)
    dyn_majority : lớp đa số của dữ liệu TRƯỚC segment
                   (val ← train; test ← train ∪ val). pct_pos>50% nên thường = +1
                   (trùng always_pos) — giữ cho đủ rào, Step 12 ghi nhận trùng.
    always_pos   : luôn +1

Output long format: ``date | k | segment | y_true | persistence | dyn_majority | always_pos``
  - ``segment ∈ {val, test}``; căn ĐÚNG (date,k,segment) với predictions_model.parquet.
  - ``y_true`` giữ NaN ở đuôi test k phiên (loại khi đánh giá ở Step 12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.split import HORIZONS, make_split

PRED_COLS = ("persistence", "dyn_majority", "always_pos")
CANONICAL = ["date", "k", "segment", "y_true", *PRED_COLS]


def _majority(y: np.ndarray) -> float:
    """Lớp đa số (bỏ NaN); tie/không xác định → +1 (quy ước dự án tie→+1)."""
    mean = np.nanmean(y)
    if not np.isfinite(mean):
        return 1.0
    return 1.0 if mean >= 0 else -1.0


def build_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Ráp bảng baseline long format trên val + test mỗi horizon."""
    dates = df["date"].to_numpy()
    parts = []
    for k in HORIZONS:
        sp = make_split(df["date"], k)
        yk = df[f"y_{k}"]
        pers_full = yk.shift(k).to_numpy()                 # nhãn dịch k dòng

        maj_val = _majority(yk.iloc[sp.train_idx].to_numpy())
        pre_test = np.concatenate([sp.train_idx, sp.val_idx])
        maj_test = _majority(yk.iloc[pre_test].to_numpy())

        for seg, idx, maj in (("val", sp.val_idx, maj_val),
                              ("test", sp.test_idx, maj_test)):
            parts.append(pd.DataFrame({
                "date": dates[idx],
                "k": k,
                "segment": seg,
                "y_true": yk.to_numpy()[idx],
                "persistence": pers_full[idx],
                "dyn_majority": float(maj),
                "always_pos": 1.0,
            }))

    out = pd.concat(parts, ignore_index=True)
    out["k"] = out["k"].astype("int64")
    return out[CANONICAL]


def validate(out: pd.DataFrame, df: pd.DataFrame) -> None:
    """4 check hợp đồng chất lượng; fail → raise (runner bắt, exit 1)."""
    # 1. schema
    if list(out.columns) != CANONICAL:
        raise ValueError(f"[baselines] schema sai: {list(out.columns)}")

    # 2. domain: dự đoán ∈ {-1,+1}, không NaN (persistence trên val/test luôn quan sát được)
    for c in PRED_COLS:
        v = out[c].to_numpy(dtype=float)
        if np.isnan(v).any():
            raise ValueError(f"[baselines] {c} có NaN")
        if not np.isin(v, (-1.0, 1.0)).all():
            raise ValueError(f"[baselines] {c} có giá trị ngoài {{-1,+1}}")

    # 3. coverage: đúng số phiên val + test mỗi k
    for k in HORIZONS:
        sp = make_split(df["date"], k)
        for seg, idx in (("val", sp.val_idx), ("test", sp.test_idx)):
            n = int(((out["k"] == k) & (out["segment"] == seg)).sum())
            if n != len(idx):
                raise ValueError(f"[baselines] k={k} {seg}: {n} dòng != {len(idx)}")

    # 4. persistence khớp y_k.shift(k) (kiểm vài điểm đầu mỗi nhóm)
    for k in HORIZONS:
        sp = make_split(df["date"], k)
        ref = df[f"y_{k}"].shift(k).to_numpy()
        sub = out[(out["k"] == k) & (out["segment"] == "test")].head(5)
        pos = {pd.Timestamp(d): i for i, d in enumerate(df["date"])}
        for _, row in sub.iterrows():
            i = pos[pd.Timestamp(row["date"])]
            if not np.isclose(row["persistence"], ref[i]):
                raise ValueError(f"[baselines] persistence lệch shift(k) tại {row['date']}")