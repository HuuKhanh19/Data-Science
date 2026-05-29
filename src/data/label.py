"""src/data/label.py — logic thuần Bước 6 (gán nhãn).

y_{t,k} = sign(P_{t+k} - P_t), k ∈ {1,5,10,20}, P = adjusted close (cột `close`).
Quy ước: tie (P_{t+k}=P_t) -> +1; k phiên cuối mỗi horizon -> NaN (giữ dòng cho
inference live, loại khỏi train). Đây là bước DUY NHẤT được nhìn tương lai.
DataFrame in -> out, không I/O (kiến trúc lai, nối tiếp integrate.py / features.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS: tuple[int, ...] = (1, 5, 10, 20)
LABELS: list[str] = [f"y_{k}" for k in HORIZONS]
PRICE_COL = "close"  # adjusted close trên spine HOSE (Bước 4: non-null toàn spine)

# Snapshot Phase 1 (29/05/2026): spine 1994 phiên -> tie counts đối chiếu EDA Phase 0.
# Chỉ dùng làm guard regression khi đúng spine khóa; data lớn lên thì tự nới (xem validate).
_SPINE_LOCKED = 1994
_TIE_COUNTS_LOCKED = {1: 155, 5: 46, 10: 21, 20: 11}


def build_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """panel (Bước 4: có `date` + `close` trên spine) -> `date` + 4 nhãn float64."""
    df = panel.sort_values("date").reset_index(drop=True)
    P = df[PRICE_COL]
    out = pd.DataFrame({"date": df["date"].to_numpy()})
    for k in HORIZONS:
        diff = P.shift(-k) - P                       # P_{t+k} - P_t; đuôi k phiên -> NaN
        y = np.where(diff >= 0, 1.0, -1.0)           # tie (diff==0) gộp vào +1
        y = np.where(diff.isna().to_numpy(), np.nan, y)  # NaN giữ NaN (không map nhầm +1)
        out[f"y_{k}"] = y.astype("float64")
    return out


def tie_counts(panel: pd.DataFrame) -> dict[int, int]:
    """Đếm ties P_{t+k}=P_t mỗi horizon (vùng có nhãn) — để log/đối chiếu EDA."""
    P = panel.sort_values("date").reset_index(drop=True)[PRICE_COL]
    n = len(panel)
    return {k: int(((P.shift(-k) - P).iloc[: n - k] == 0).sum()) for k in HORIZONS}


def validate(labels: pd.DataFrame, panel: pd.DataFrame) -> None:
    """4-check contract (hợp đồng chất lượng). Fail -> raise ValueError."""
    _check_spine_aligned(labels, panel)
    _check_label_domain(labels)
    _check_tail_nan_exact(labels)
    _check_tie_convention(labels, panel)


def _check_spine_aligned(labels: pd.DataFrame, panel: pd.DataFrame) -> None:
    d = labels["date"]
    if len(labels) != len(panel):
        raise ValueError(f"[spine_aligned] {len(labels)} dòng != panel {len(panel)}")
    if not d.is_monotonic_increasing or bool(d.duplicated().any()):
        raise ValueError("[spine_aligned] date không tăng nghiêm ngặt / có trùng")
    if not d.reset_index(drop=True).equals(panel["date"].reset_index(drop=True)):
        raise ValueError("[spine_aligned] date không trùng khớp spine panel")


def _check_label_domain(labels: pd.DataFrame) -> None:
    for c in LABELS:
        vals = set(pd.unique(labels[c].dropna()))
        extra = vals - {-1.0, 1.0}
        if extra:
            raise ValueError(f"[label_domain] {c} có giá trị lạ: {sorted(extra)}")


def _check_tail_nan_exact(labels: pd.DataFrame) -> None:
    n = len(labels)
    for k in HORIZONS:
        col = labels[f"y_{k}"]
        n_nan = int(col.isna().sum())
        if n_nan != k:
            raise ValueError(f"[tail_nan_exact] y_{k}: {n_nan} NaN != k={k}")
        if not col.iloc[n - k :].isna().all() or not col.iloc[: n - k].notna().all():
            raise ValueError(f"[tail_nan_exact] y_{k}: NaN không nằm trọn ở đuôi")


def _check_tie_convention(labels: pd.DataFrame, panel: pd.DataFrame) -> None:
    P = panel.sort_values("date").reset_index(drop=True)[PRICE_COL]
    n = len(panel)
    counts: dict[int, int] = {}
    for k in HORIZONS:
        diff = (P.shift(-k) - P).iloc[: n - k]       # vùng có nhãn
        tie_mask = (diff == 0).to_numpy()
        counts[k] = int(tie_mask.sum())
        y_at_tie = labels[f"y_{k}"].iloc[: n - k].to_numpy()[tie_mask]
        if not bool((y_at_tie == 1.0).all()):
            raise ValueError(f"[tie_convention] y_{k}: có tie không gán +1")
    # Snapshot guard: chỉ assert số đếm khi ĐÚNG spine khóa Phase 1 (1994 phiên).
    # Data lớn lên (live/Phase 2) -> bỏ qua đối chiếu cứng, vẫn giữ invariant tie->+1 ở trên.
    if n == _SPINE_LOCKED and counts != _TIE_COUNTS_LOCKED:
        raise ValueError(f"[tie_convention] tie counts {counts} != EDA khóa {_TIE_COUNTS_LOCKED}")