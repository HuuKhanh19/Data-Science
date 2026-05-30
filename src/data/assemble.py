"""src/data/assemble.py — logic thuần Step 7 (lắp ráp, xử lý NA, xuất artifact).

Merge `features_raw` (Step 5) + `labels` (Step 6) theo `date`, cắt vùng warmup đầu
chuỗi (TỰ SUY từ NaN feature, không đọc log), giữ tail-NaN nhãn cho inference live.
Ra `date` + 20 feature + 4 nhãn (đúng thứ tự canonical). KHÔNG PCA / KHÔNG feature
selection — confirmatory. DataFrame in -> out, không I/O.

FEATURES / LABELS import từ module nguồn (single source of truth).
"""
from __future__ import annotations

import pandas as pd

from src.data.features import FEATURES
from src.data.label import HORIZONS, LABELS

CANONICAL = ["date", *FEATURES, *LABELS]  # 1 + 20 + 4 = 25 cột


def assemble(features_raw: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Ráp 2 mảnh interim -> dataset đã cắt warmup. Tự suy mốc cắt từ NaN feature."""
    if len(features_raw) != len(labels):
        raise ValueError(
            f"[assemble] spine lệch: features_raw {len(features_raw)} != labels {len(labels)}"
        )
    df = features_raw.merge(labels, on="date", how="inner", validate="one_to_one")
    if len(df) != len(features_raw):
        raise ValueError(
            f"[assemble] merge rớt dòng: {len(df)} != {len(features_raw)} (date không khớp)"
        )
    df = df.sort_values("date").reset_index(drop=True)

    # Cắt warmup: dòng ĐẦU TIÊN mà cả 20 feature đều non-NaN (tự suy).
    full = df[FEATURES].notna().all(axis=1)
    if not bool(full.any()):
        raise ValueError("[assemble] không có dòng nào đủ 20 feature non-NaN")
    start = int(full.idxmax())  # vị trí đầu tiên True (index đã reset 0..n-1)
    out = df.iloc[start:].reset_index(drop=True)

    return out[CANONICAL]


def validate(out: pd.DataFrame) -> None:
    """4-check contract. Fail -> raise ValueError."""
    _check_schema(out)
    _check_features_no_nan(out)
    _check_features_finite(out)
    _check_label_tail_nan(out)
    _check_date_key(out)


def _check_schema(out: pd.DataFrame) -> None:
    if list(out.columns) != CANONICAL:
        raise ValueError(
            f"[schema] cột sai. Kỳ vọng {len(CANONICAL)} cột đúng thứ tự "
            f"date + {len(FEATURES)} feature + {len(LABELS)} nhãn"
        )


def _check_features_no_nan(out: pd.DataFrame) -> None:
    n_nan = int(out[FEATURES].isna().to_numpy().sum())
    if n_nan != 0:
        bad = {c: int(out[c].isna().sum()) for c in FEATURES if out[c].isna().any()}
        raise ValueError(f"[features_no_nan] còn NaN trong vùng khả dụng: {bad}")


def _check_features_finite(out: pd.DataFrame) -> None:
    """inf KHÁC NaN — no_nan không bắt được. Chặn ±inf lọt vào model (vd P/0)."""
    import numpy as np
    arr = out[FEATURES].to_numpy(dtype=float)
    if np.isinf(arr).any():
        bad = {c: int(np.isinf(out[c].to_numpy(dtype=float)).sum())
               for c in FEATURES if np.isinf(out[c].to_numpy(dtype=float)).any()}
        raise ValueError(f"[features_finite] còn ±inf: {bad}")


def _check_label_tail_nan(out: pd.DataFrame) -> None:
    n = len(out)
    for k in HORIZONS:
        col = out[f"y_{k}"]
        n_nan = int(col.isna().sum())
        if n_nan != k:
            raise ValueError(f"[label_tail_nan] y_{k}: {n_nan} NaN != k={k}")
        if not col.iloc[n - k :].isna().all() or not col.iloc[: n - k].notna().all():
            raise ValueError(f"[label_tail_nan] y_{k}: NaN không nằm trọn ở đuôi")


def _check_date_key(out: pd.DataFrame) -> None:
    d = out["date"]
    if bool(d.duplicated().any()):
        raise ValueError("[date_key] date có trùng lặp")
    if not d.is_monotonic_increasing:
        raise ValueError("[date_key] date không tăng nghiêm ngặt")