"""Step 9 — Baselines (TCB direction).

Sinh dự đoán 3 baseline tầm thường trên ĐÚNG lịch test walk-forward (Step 8),
làm RÀO để DM test ở Step 12 so với model. Không học, không GPU — thuần quy tắc.

    persistence  : ŷ_{t,k} = sign(P_t − P_{t-k}) = nhãn đã hiện thực hóa k phiên trước
                   = y_{·,k}.shift(k)  (không cần giá thô)
    dyn_majority : lớp đa số trong train window cuối (refit hàng tuần, dùng splitter Step 8)
    always_pos   : luôn +1

Output: long format (date, k, y_true, persistence, dyn_majority, always_pos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.walk_forward import walk_forward_splits

HORIZONS = (1, 5, 10, 20)  # gương theo src/data/label.py
PRED_COLS = ("persistence", "dyn_majority", "always_pos")
CANONICAL = ["date", "k", "y_true", *PRED_COLS]


def _majority(y: np.ndarray) -> float:
    """Lớp đa số trong train window; tie (đếm bằng nhau) → +1 (quy ước dự án)."""
    return 1.0 if np.nanmean(y) >= 0 else -1.0


def build_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Ráp bảng baseline long format trên các phiên test walk-forward của từng k."""
    parts = []
    for k in HORIZONS:
        yk = df[f"y_{k}"]
        pers_full = yk.shift(k)                      # persistence = nhãn dịch k dòng
        dynmaj = pd.Series(np.nan, index=df.index)   # điền theo từng tuần refit
        test_chunks = []
        for train_idx, test_idx in walk_forward_splits(df["date"], k):
            dynmaj.iloc[test_idx] = _majority(yk.iloc[train_idx].to_numpy())
            test_chunks.append(test_idx)
        idx = np.concatenate(test_chunks)            # mọi phiên test của horizon k
        parts.append(pd.DataFrame({
            "date": df["date"].to_numpy()[idx],
            "k": k,
            "y_true": yk.to_numpy()[idx],
            "persistence": pers_full.to_numpy()[idx],
            "dyn_majority": dynmaj.to_numpy()[idx],
            "always_pos": 1.0,
        }))
    out = pd.concat(parts, ignore_index=True)
    out["k"] = out["k"].astype("int64")
    return out[CANONICAL]


def validate(out: pd.DataFrame, df: pd.DataFrame) -> None:
    """4 check hợp đồng chất lượng output; fail → raise."""
    # 1. schema — đúng cột + thứ tự + dtype
    assert list(out.columns) == CANONICAL, f"schema lệch: {list(out.columns)}"
    assert out["k"].dtype.kind == "i", "k phải integer"
    for c in PRED_COLS:
        assert out[c].dtype.kind == "f", f"{c} phải float"

    # 2. pred_domain — 3 cột dự đoán ∈ {-1,+1}, không NaN; always_pos toàn +1
    for c in PRED_COLS:
        assert out[c].notna().all(), f"{c} có NaN"
        assert set(np.unique(out[c])) <= {-1.0, 1.0}, f"{c} ngoài miền ±1"
    assert (out["always_pos"] == 1.0).all(), "always_pos không toàn +1"

    # 3. coverage — (date,k) khớp ĐÚNG lịch walk-forward, không trùng
    assert not out.duplicated(["date", "k"]).any(), "trùng (date,k)"
    assert set(out["k"].unique()) == set(HORIZONS), "tập k sai"
    for k in HORIZONS:
        want = sorted(
            pd.Timestamp(d)
            for _, te in walk_forward_splits(df["date"], k)
            for d in df["date"].to_numpy()[te]
        )
        got = sorted(pd.Timestamp(d) for d in out.loc[out["k"] == k, "date"])
        assert got == want, f"coverage k={k} lệch lịch walk-forward"

    # 4. y_true_tail_nan — NaN của y_true mỗi k đúng = k và nằm trọn ở đuôi (theo date)
    for k in HORIZONS:
        nan_mask = out.loc[out["k"] == k].sort_values("date")["y_true"].isna().to_numpy()
        assert nan_mask.sum() == k, f"y_true k={k}: {int(nan_mask.sum())} NaN, chờ {k}"
        assert nan_mask[-k:].all() and not nan_mask[:-k].any(), f"NaN k={k} không ở đuôi"