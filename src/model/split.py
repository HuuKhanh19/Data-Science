"""Step 7 — Chia tập 80:10:10 + embargo k (TCB direction, Framework A).

THAY cho ``walk_forward.py`` cũ (rolling window=1000 + vòng refit). Bỏ refit ở
Phase 2 nên một split thời gian cố định là đủ và đơn giản hơn:

    train (80% cũ nhất)   -> fit
    val   (10% giữa)      -> tune + chọn model + chọn feature_set
    test  (10% mới nhất)  -> chạm MỘT LẦN

Module logic THUẦN (không I/O bền). Step này chỉ ĐỊNH NGHĨA CHỈ SỐ train/val/test.
Việc ÁP DỤNG (fit Z-score trên train rồi apply val/test) nằm ở Step 11.

Anti-leakage 3/4 — embargo ``k`` ở mỗi ranh giới:
  nhãn ``y_{t,k}`` nhìn ``k`` phiên tới, nên ``k`` dòng cuối mỗi đoạn (train, val)
  bị loại để nhãn không rỉ sang đoạn sau. Test giữ nguyên; ``k`` dòng cuối toàn
  chuỗi có nhãn NaN (giữ cho inference live, loại khi đánh giá ở Step 12).

Ranh giới train|val và val|test là k-ĐỘC LẬP; chỉ phần đuôi bị embargo phụ thuộc
``k`` -> kích thước train/val giảm đúng ``k`` dòng theo từng horizon.

Giữ 3 helper dùng chung cho tầng model (Step 9-11):
  ``feature_columns`` / ``drop_label_nan`` / ``scale_train_test`` (chốt leakage 4/4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HORIZONS = (1, 5, 10, 20)
LABEL_COLS = ("y_1", "y_5", "y_10", "y_20")
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# TEST_FRAC = phần còn lại (train nhận phần dư để dùng hết dòng).


# ──────────────────────────── helper dùng chung ────────────────────────────
def feature_columns(df: pd.DataFrame) -> list[str]:
    """Cột feature = mọi cột trừ ``date`` và 4 nhãn (toàn bộ pool ứng viên).

    Việc chọn bộ con (l1/eda/full) là ở tầng model — split feature-agnostic.
    """
    return [c for c in df.columns if c not in ("date", *LABEL_COLS)]


def drop_label_nan(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Loại dòng có nhãn NaN (đuôi chuỗi mỗi horizon). An toàn cho mọi tập."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    return X[mask], y[mask]


def scale_train_test(X_fit, *X_apply):
    """Z-score: fit μ,σ CHỈ trên ``X_fit`` (= train) rồi apply cho train + mọi tập khác.

    Trả ``(X_fit_z, *X_apply_z, scaler)``. Chốt leakage 4/4 — chuẩn hóa không nhìn
    val/test. Gọi được với 1 tập (train,test) như cũ hoặc nhiều tập (train,val,test).
    """
    scaler = StandardScaler().fit(np.asarray(X_fit, dtype=float))
    out = [scaler.transform(np.asarray(X_fit, dtype=float))]
    out += [scaler.transform(np.asarray(x, dtype=float)) for x in X_apply]
    out.append(scaler)
    return tuple(out)


# ──────────────────────────── split ────────────────────────────
@dataclass(frozen=True)
class Split:
    """Chỉ số 3 tập cho 1 horizon ``k`` (đã embargo)."""
    k: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_end: int   # ranh giới train|val (vị trí, trước embargo) — k-độc lập
    val_end: int     # ranh giới val|test (vị trí, trước embargo) — k-độc lập
    n: int


def make_split(dates, k: int,
               train_frac: float = TRAIN_FRAC,
               val_frac: float = VAL_FRAC) -> Split:
    """Split 80:10:10 theo thời gian + embargo ``k``. Trả chỉ số vị trí (0..n-1).

        train = [0, train_end - k)
        val   = [train_end, val_end - k)
        test  = [val_end, n)

    với ``train_end = n_train``, ``val_end = n_train + n_val``.
    """
    dt = pd.DatetimeIndex(pd.to_datetime(dates))
    if not (dt.is_monotonic_increasing and dt.is_unique):
        raise ValueError("dates phải tăng nghiêm ngặt và không trùng")
    if k <= 0:
        raise ValueError(f"k phải > 0, nhận {k}")

    n = len(dt)
    n_test = round((1.0 - train_frac - val_frac) * n)
    n_val = round(val_frac * n)
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) <= k:
        raise ValueError(
            f"k={k} quá lớn so với kích thước tập "
            f"(train={n_train}, val={n_val}, test={n_test})"
        )

    train_end = n_train
    val_end = n_train + n_val
    train_idx = np.arange(0, train_end - k)
    val_idx = np.arange(train_end, val_end - k)
    test_idx = np.arange(val_end, n)
    return Split(k, train_idx, val_idx, test_idx, train_end, val_end, n)


def split_summary(df: pd.DataFrame, k: int) -> dict:
    """Metadata 1 horizon cho log: kích thước, mốc ngày, số nhãn dùng được."""
    sp = make_split(df["date"], k)
    dates = pd.to_datetime(df["date"]).to_numpy()
    yk = df[f"y_{k}"].to_numpy(dtype=float)

    def seg(idx: np.ndarray) -> dict:
        n_label = int(np.sum(~np.isnan(yk[idx]))) if len(idx) else 0
        return {
            "n": int(len(idx)),
            "n_label": n_label,                  # nhãn không NaN (loại đuôi inference)
            "n_independent": int(n_label // k),  # nhãn không chồng lấp (xấp xỉ, do overlapping)
            "pct_pos": (round(float(np.nanmean(yk[idx] == 1) * 100), 2)
                        if n_label else None),
            "date_start": str(pd.Timestamp(dates[idx[0]]).date()) if len(idx) else None,
            "date_end": str(pd.Timestamp(dates[idx[-1]]).date()) if len(idx) else None,
        }

    return {
        "k": k,
        "embargo": k,
        "train": seg(sp.train_idx),
        "val": seg(sp.val_idx),
        "test": seg(sp.test_idx),
    }


def validate(df: pd.DataFrame) -> None:
    """Hợp đồng split: 3 tập rời nhau, đúng thứ tự thời gian, embargo đúng ``k``.

    Fail -> raise ValueError (runner bắt, exit 1).
    """
    dt = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    if not (dt.is_monotonic_increasing and dt.is_unique):
        raise ValueError("[split] date không tăng nghiêm ngặt / có trùng")

    for k in HORIZONS:
        sp = make_split(df["date"], k)
        tr, va, te = sp.train_idx, sp.val_idx, sp.test_idx

        # 1. không tập nào rỗng
        if min(len(tr), len(va), len(te)) == 0:
            raise ValueError(f"[split] k={k}: có tập rỗng")
        # 2. đúng thứ tự thời gian + rời nhau
        if not (tr[-1] < va[0] and va[-1] < te[0]):
            raise ValueError(f"[split] k={k}: tập không rời / sai thứ tự")
        # 3. embargo đúng k ở mỗi ranh giới
        gap_tv = va[0] - tr[-1] - 1
        gap_vt = te[0] - va[-1] - 1
        if gap_tv != k:
            raise ValueError(f"[split] k={k}: gap train|val = {gap_tv} != {k}")
        if gap_vt != k:
            raise ValueError(f"[split] k={k}: gap val|test = {gap_vt} != {k}")
        # 4. test chạm hết đuôi chuỗi (giữ dòng inference live)
        if te[-1] != sp.n - 1:
            raise ValueError(f"[split] k={k}: test không chạm cuối chuỗi")
        # 5. ranh giới k-độc lập (sanity)
        if sp.train_end != make_split(df["date"], 1).train_end:
            raise ValueError(f"[split] k={k}: ranh giới train|val không k-độc lập")


if __name__ == "__main__":  # đối chiếu nhanh khi chạy thẳng module
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.parquet"
    df = pd.read_parquet(path)
    validate(df)
    dates = pd.to_datetime(df["date"])
    sp1 = make_split(df["date"], 1)
    print(f"features.parquet: {len(df)} phiên | "
          f"{dates.iloc[0].date()} -> {dates.iloc[-1].date()}")
    print(f"ranh giới train|val @ idx {sp1.train_end} ({dates.iloc[sp1.train_end].date()}) "
          f"| val|test @ idx {sp1.val_end} ({dates.iloc[sp1.val_end].date()})")
    for k in HORIZONS:
        s = split_summary(df, k)
        print(f"  k={k:<2d} train={s['train']['n']:<5d} val={s['val']['n']:<4d} "
              f"test={s['test']['n']:<4d} | test nhãn={s['test']['n_label']} "
              f"(độc lập ~{s['test']['n_independent']})")