"""Step 8 — Cầu nối walk-forward + Z-score per-window (TCB direction).

Module logic THUẦN, không I/O bền: biến ``features.parquet`` thành các cặp
(train, test) per-window phù du trong RAM, dùng trong vòng lặp refit ở Step 11.
Đóng 2 chốt leakage còn lại:

    (3) buffer gap ``k`` ở cuối train   -> ``walk_forward_splits``
    (4) Z-score fit CHỈ trên train      -> ``scale_train_test``

Splitter thuần index (không đụng nhãn); lọc NaN + chuẩn hóa làm ở ``build_window``.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

N_TRAIN = 1000
LABEL_COLS = ("y_1", "y_5", "y_10", "y_20")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """20 cột feature = mọi cột trừ ``date`` và 4 nhãn."""
    return [c for c in df.columns if c not in ("date", *LABEL_COLS)]


def iso_week_blocks(dates) -> list[np.ndarray]:
    """Gom các phiên liên tiếp theo tuần lịch ISO -> list block chỉ số vị trí.

    Mỗi block = các dòng cùng (iso_year, iso_week). Tuần có ngày nghỉ -> block < 5.
    Yêu cầu ``dates`` tăng dần (đảm bảo bởi ``date_key`` check ở Step 7).
    """
    dt = pd.DatetimeIndex(pd.to_datetime(dates))
    if not dt.is_monotonic_increasing:
        raise ValueError("dates phải tăng dần")
    iso = dt.isocalendar()
    keys = list(zip(iso["year"].to_numpy(), iso["week"].to_numpy()))
    blocks: list[np.ndarray] = []
    start, n = 0, len(keys)
    for i in range(1, n + 1):
        if i == n or keys[i] != keys[start]:
            blocks.append(np.arange(start, i))
            start = i
    return blocks


def walk_forward_splits(
    dates, k: int, n_train: int = N_TRAIN
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Sinh (train_idx, test_idx) theo vị trí: rolling ``n_train`` phiên + buffer gap ``k``.

    Test block = 1 tuần lịch ISO. Với tuần có phiên đầu ở vị trí ``t0``::

        train = [t0 - n_train - k, t0 - k)   # đúng n_train phiên
        gap   = [t0 - k, t0)                  # k phiên bị loại — nhãn chưa quan sát
        test  = các phiên của tuần đó         # bắt đầu ở t0

    Chỉ phát những tuần có train đầy đủ (``t0 >= n_train + k``) -> tuần đầu T_w≈n_train+k.
    """
    dt = pd.DatetimeIndex(pd.to_datetime(dates))
    if not (dt.is_monotonic_increasing and dt.is_unique):
        raise ValueError("dates phải tăng nghiêm ngặt và không trùng")
    for test_idx in iso_week_blocks(dt):
        t0 = int(test_idx[0])
        train_start = t0 - n_train - k
        if train_start < 0:
            continue
        train_idx = np.arange(train_start, t0 - k)  # đúng n_train phiên
        yield train_idx, test_idx


def drop_label_nan(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Loại dòng có nhãn NaN (an toàn; buffer gap đã đảm bảo train sạch NaN)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    return X[mask], y[mask]


def scale_train_test(X_train, X_test) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Z-score: fit μ,σ CHỈ trên train rồi apply cho cả train & test (chốt leakage 4)."""
    scaler = StandardScaler().fit(np.asarray(X_train, dtype=float))
    x_tr = scaler.transform(np.asarray(X_train, dtype=float))
    x_te = scaler.transform(np.asarray(X_test, dtype=float))
    return x_tr, x_te, scaler


def build_window(df, train_idx, test_idx, label_col, feature_cols=None):
    """Ráp một window: chọn cột -> lọc NaN nhãn train -> z-score -> trả ma trận RAM.

    Trả: ``(X_train_z, y_train, X_test_z, y_test, test_dates, scaler)``.
    ``y_test`` có thể chứa NaN ở đuôi chuỗi (giữ cho inference; loại khi đánh giá).
    """
    if feature_cols is None:
        feature_cols = feature_columns(df)
    x_tr = df.iloc[train_idx][feature_cols].to_numpy(dtype=float)
    y_tr = df.iloc[train_idx][label_col].to_numpy(dtype=float)
    x_te = df.iloc[test_idx][feature_cols].to_numpy(dtype=float)
    y_te = df.iloc[test_idx][label_col].to_numpy(dtype=float)
    x_tr, y_tr = drop_label_nan(x_tr, y_tr)
    x_tr_z, x_te_z, scaler = scale_train_test(x_tr, x_te)
    test_dates = pd.DatetimeIndex(pd.to_datetime(df.iloc[test_idx]["date"]))
    return x_tr_z, y_tr, x_te_z, y_te, test_dates, scaler


def phase0_boundary_date(dates, n_phase0: int = N_TRAIN) -> pd.Timestamp:
    """Ngày phiên đầu tiên SAU Phase-0 (vị trí index ``n_phase0``).

    Phase-0 = ``[0, n_phase0)``; đây là mốc ranh giới Phase-0/test cần ghi lại
    (đối chiếu một lần với research_design §3.1 rồi khóa bất biến).
    """
    dt = pd.DatetimeIndex(pd.to_datetime(dates))
    return dt[n_phase0]


if __name__ == "__main__":  # đối chiếu một lần: in mốc Phase-0/test + số tuần refit
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.parquet"
    df = pd.read_parquet(path)
    dates = df["date"]
    print(f"features.parquet: {len(df)} phiên | "
          f"{pd.Timestamp(dates.iloc[0]).date()} -> {pd.Timestamp(dates.iloc[-1]).date()}")
    print(f"Phase-0 phiên cuối  (index {N_TRAIN - 1}): "
          f"{pd.Timestamp(dates.iloc[N_TRAIN - 1]).date()}")
    print(f"Test phiên đầu (mốc) (index {N_TRAIN}): "
          f"{pd.Timestamp(phase0_boundary_date(dates)).date()}")
    for k in (1, 5, 10, 20):
        splits = list(walk_forward_splits(dates, k))
        first_te = pd.Timestamp(dates.iloc[splits[0][1][0]]).date()
        last_te = pd.Timestamp(dates.iloc[splits[-1][1][-1]]).date()
        print(f"  k={k:<2d}: {len(splits):3d} tuần refit | test {first_te} -> {last_te}")