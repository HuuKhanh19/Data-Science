"""Unit test Step 8 — walk-forward splitter + buffer gap + z-score per-window.

Chạy từ gốc repo: ``python -m pytest tests/test_walk_forward.py -v``
"""
import numpy as np
import pandas as pd
import pytest

from src.model.walk_forward import (
    N_TRAIN,
    iso_week_blocks,
    walk_forward_splits,
    scale_train_test,
    drop_label_nan,
    build_window,
    feature_columns,
    phase0_boundary_date,
)


def make_df(n_days=1300, n_feat=4, seed=0):
    """Chuỗi giả lập: n_days phiên (business day), bỏ vài ngày để tạo tuần ngắn < 5."""
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range("2018-01-01", periods=n_days + 50)
    drop = {bdays[100], bdays[101], bdays[250], bdays[600]}  # vài holiday rải rác
    dates = pd.DatetimeIndex([d for d in bdays if d not in drop][:n_days])
    df = pd.DataFrame({"date": dates})
    for j in range(n_feat):  # feature ramp tuyến tính + nhiễu nhỏ -> phân phối train/test lệch
        df[f"f{j}"] = np.arange(n_days, dtype=float) + rng.normal(0, 0.01, n_days)
    for k in (1, 5, 10, 20):  # nhãn ±1, đuôi đúng k phiên là NaN
        y = rng.choice([-1.0, 1.0], size=n_days)
        y[-k:] = np.nan
        df[f"y_{k}"] = y
    return df


@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_train_length_and_buffer_gap(k):
    df = make_df()
    splits = list(walk_forward_splits(df["date"], k))
    assert len(splits) > 0
    for tr, te in splits:
        assert len(tr) == N_TRAIN                                   # train đúng 1000 phiên
        assert np.array_equal(tr, np.arange(tr[0], tr[0] + N_TRAIN))  # liền mạch
        assert te[0] - tr[-1] == k + 1                              # đúng k phiên bị bỏ
        assert tr.max() < te.min()                                  # no leakage


@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_first_week_after_phase0(k):
    df = make_df()
    splits = list(walk_forward_splits(df["date"], k))
    first_t0 = int(splits[0][1][0])
    assert first_t0 >= N_TRAIN + k                                  # train đầu phải đủ chỗ
    starts = [int(b[0]) for b in iso_week_blocks(df["date"])]
    emitted = [s for s in starts if s - N_TRAIN - k >= 0]
    assert emitted[0] == first_t0                                   # đúng tuần đầu đủ điều kiện
    idx = starts.index(first_t0)
    if idx > 0:
        assert starts[idx - 1] - N_TRAIN - k < 0                   # tuần trước đó bị loại đúng


def test_each_test_block_is_one_iso_week():
    df = make_df()
    iso = pd.DatetimeIndex(df["date"]).isocalendar()
    yr, wk = iso["year"].to_numpy(), iso["week"].to_numpy()
    for _, te in walk_forward_splits(df["date"], 5):
        assert len(set(zip(yr[te], wk[te]))) == 1                   # mỗi block đúng 1 tuần lịch


def test_test_blocks_disjoint_and_increasing():
    df = make_df()
    prev_max = -1
    for _, te in walk_forward_splits(df["date"], 5):
        assert te.min() > prev_max
        prev_max = int(te.max())


def test_iso_week_short_week_exists():
    sizes = [len(b) for b in iso_week_blocks(make_df()["date"])]
    assert max(sizes) <= 5
    assert min(sizes) < 5                                           # có tuần ngắn do holiday


def test_zscore_fit_on_train_only():
    df = make_df()
    splits = list(walk_forward_splits(df["date"], 5))
    tr, te = splits[len(splits) // 2]                              # 1 window giữa chuỗi
    feats = feature_columns(df)
    x_tr = df.iloc[tr][feats].to_numpy(float)
    x_te = df.iloc[te][feats].to_numpy(float)
    x_tr_z, x_te_z, scaler = scale_train_test(x_tr, x_te)
    assert np.allclose(scaler.mean_, x_tr.mean(axis=0))                       # μ = thống kê train
    assert not np.allclose(scaler.mean_, df[feats].to_numpy(float).mean(0))   # KHÔNG phải toàn chuỗi
    assert np.allclose(x_tr_z.mean(axis=0), 0, atol=1e-7)                     # train ~ N(0,1)
    assert np.allclose(x_tr_z.std(axis=0), 1, atol=1e-6)
    assert (x_te_z.mean(axis=0) > 1).all()                                    # test (ramp lớn hơn) không bị ép về 0


def test_build_window_train_has_no_nan_labels():
    df = make_df()
    tr, te = list(walk_forward_splits(df["date"], 20))[-1]          # window gần đuôi nhất
    x_tr_z, y_tr, x_te_z, y_te, test_dates, _ = build_window(df, tr, te, "y_20")
    assert not np.isnan(y_tr).any()                                # gap đảm bảo train sạch NaN
    assert len(y_tr) == len(x_tr_z)
    assert len(test_dates) == len(te)


def test_drop_label_nan():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([1, -1, np.nan, 1, np.nan, -1, 1, 1, -1, np.nan])
    Xf, yf = drop_label_nan(X, y)
    assert len(yf) == 7 and not np.isnan(yf).any() and Xf.shape == (7, 2)


def test_phase0_boundary_date():
    df = make_df()
    assert phase0_boundary_date(df["date"]) == pd.Timestamp(df["date"].iloc[N_TRAIN])


def test_rejects_unsorted_dates():
    bad = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-02"]))
    with pytest.raises(ValueError):
        list(walk_forward_splits(bad, 1))