"""
src/data/integrate.py — Bước 4 (Phase 1, Step 2): Tích hợp dữ liệu.

Logic THUẦN (DataFrame in -> DataFrame out), không I/O. Runner mỏng ở
`scripts/integrate_phase1.py` lo đọc/ghi file.

Gộp 6 nguồn (sau Bước 3) thành MỘT panel daily trên trục spine HOSE, trong đó
mỗi dòng t phản ánh đúng trạng thái thông tin ĐÃ CÔNG BỐ tính đến hết ngày t.
Đầu ra vẫn là dữ liệu THÔ (chưa YoY/log-return — để dành Bước 5).

Hai cơ chế ghép khác BẢN CHẤT:
  1. Daily merge — VNINDEX & USD/VND cùng tần suất, đã trên spine (Bước 3)
     => ghép theo `date` (độ an toàn cao, không lệch tần suất).
  2. As-of join  — CPI/GDP/fundamentals trễ release => merge_asof BACKWARD theo
     `release_date <= t`. Đây là CHỐT CHẶN LEAKAGE 2/4. Forward-fill giữa hai kỳ
     release là BẢN CHẤT của backward join (không phải bước rời).

Hai loại forward-fill, đừng nhầm:
  - ffill GIỮA CHUỖI KỲ (hàm prep_slow): lấp field thiếu *trong* nguồn chậm,
    vd fundamentals 2021-Q2 NPL/NIM lấy từ 2021-Q1 (an toàn leakage; Q1 công bố
    trước cửa sổ Q2). NaN dẫn đầu (2018-Q1 NPL/NIM) không lấp được -> prefix.
  - ffill GIỮA RELEASE (hàm asof_join): kéo bản công bố gần nhất xuống mọi phiên
    daily tới kỳ kế — built-in của direction="backward".

Ranh giới trách nhiệm (đã chốt với data.md):
  - Bước 4 align "đã biết tới t" (same-day, release_date <= t). Lag t-1 chống
    leakage hoàn toàn là việc Bước 5.
  - YoY cũng để Bước 5: panel giữ `*_reference_period` để Bước 5 dựng lại chuỗi
    kỳ và tính YoY trên chuỗi raw (chính xác, không xấp xỉ từ daily đã ffill).
"""

from __future__ import annotations

import pandas as pd

# OHLCV của 3 nguồn daily (sau clean). usdvnd thêm cờ fx_ffilled.
_OHLCV = ["open", "high", "low", "close", "volume"]

# Cột chết: vnstock không cung cấp, 33/33 NaN -> bỏ ngay ở tích hợp (chốt EDA Phase 0).
_FUND_DROP = ["interest_earning_assets_vnd_bil"]

# Prefix cho 3 nguồn chậm (đặt cho reference_period/release_date để audit).
_SLOW_PREFIXES = ("cpi", "gdp", "fund")


# =============================================================================
# 1. Daily merge
# =============================================================================
def merge_daily(spine: pd.DataFrame, vnindex: pd.DataFrame,
                usdvnd: pd.DataFrame) -> pd.DataFrame:
    """Ghép VNINDEX & USD/VND vào spine theo `date`.

    Cả 3 đã đồng trục spine ở Bước 3 => yêu cầu TRÙNG KHỚP tập ngày, lệch là lỗi
    data (raise), không bao giờ tự sinh/điền ngày. TCB OHLCV giữ tên trần (spine
    = subject, P_t = `close`); VNINDEX/USD-VND prefix để tránh đụng cột.
    """
    spine = spine.sort_values("date").reset_index(drop=True)
    s_dates = spine["date"].reset_index(drop=True)

    out = spine[["date", *_OHLCV]].copy()  # TCB OHLCV: tên trần

    for name, df, extra in [("vnindex", vnindex, []),
                            ("usdvnd", usdvnd, ["fx_ffilled"])]:
        df = df.sort_values("date").reset_index(drop=True)
        if not df["date"].reset_index(drop=True).equals(s_dates):
            raise ValueError(
                f"{name}: tập ngày KHÔNG trùng spine (đáng lẽ đã căn ở Bước 3). "
                f"spine={len(s_dates)} vs {name}={len(df)}."
            )
        cols = _OHLCV + extra
        renamed = df[cols].rename(columns={c: f"{name}_{c}" for c in _OHLCV})
        out = pd.concat([out, renamed.reset_index(drop=True)], axis=1)

    return out


# =============================================================================
# 2. Chuẩn bị nguồn chậm + as-of join
# =============================================================================
def prep_slow(df: pd.DataFrame, drop_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Sắp theo `reference_period`, bỏ cột chết, forward-fill cột giá trị.

    ffill TRÊN CHUỖI KỲ (không phải daily) = "carry last known quarterly/monthly
    value": lấp hố field thiếu giữa chuỗi (fundamentals 2021-Q2 NPL/NIM <- Q1).
    NaN dẫn đầu (2018-Q1 NPL/NIM) không lấp được -> prefix trong panel -> cắt warmup.
    """
    df = df.sort_values("reference_period").reset_index(drop=True)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    value_cols = [c for c in df.columns
                  if c not in ("reference_period", "release_date", "fetched_at")]
    df[value_cols] = df[value_cols].ffill()
    return df


def asof_join(panel: pd.DataFrame, slow: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """As-of join backward: tại mỗi phiên t lấy bản ghi có `release_date <= t` gần nhất.

    Forward-fill giữa hai kỳ release là BẢN CHẤT của direction="backward". Giữ
    `reference_period`/`release_date` (đổi tên có prefix) để audit + để Bước 5
    tính YoY trên chuỗi kỳ. Bỏ `fetched_at`.
    """
    panel = panel.sort_values("date").reset_index(drop=True)
    slow = slow.sort_values("release_date").reset_index(drop=True)
    slow = slow.drop(columns=[c for c in ("fetched_at",) if c in slow.columns])
    slow = slow.rename(columns={
        "reference_period": f"{prefix}_reference_period",
        "release_date": f"{prefix}_release_date",
    })
    return pd.merge_asof(
        panel, slow,
        left_on="date", right_on=f"{prefix}_release_date",
        direction="backward",
    )


def integrate(spine: pd.DataFrame, vnindex: pd.DataFrame, usdvnd: pd.DataFrame,
              cpi: pd.DataFrame, gdp: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp: daily merge -> as-of join 3 nguồn chậm -> panel raw aligned."""
    panel = merge_daily(spine, vnindex, usdvnd)
    panel = asof_join(panel, prep_slow(cpi), "cpi")
    panel = asof_join(panel, prep_slow(gdp), "gdp")
    panel = asof_join(panel, prep_slow(fund, drop_cols=tuple(_FUND_DROP)), "fund")
    return panel


# =============================================================================
# 3. Kiểm định — 4 check (gương Bước 3)
# =============================================================================
def _daily_cols() -> list[str]:
    return ([*_OHLCV]
            + [f"vnindex_{c}" for c in _OHLCV]
            + [f"usdvnd_{c}" for c in _OHLCV])


def check_spine_aligned(panel: pd.DataFrame, spine: pd.DataFrame) -> None:
    """1. `date` của panel TRÙNG spine: tăng nghiêm ngặt, không trùng, đúng tập."""
    d = panel["date"]
    if not d.is_monotonic_increasing or d.duplicated().any():
        raise ValueError("panel.date không tăng nghiêm ngặt / có trùng.")
    spine_d = spine.sort_values("date")["date"].reset_index(drop=True)
    if not d.reset_index(drop=True).equals(spine_d):
        raise ValueError("panel.date lệch spine.")


def check_daily_complete(panel: pd.DataFrame) -> None:
    """2. TCB & VNINDEX OHLCV + `usdvnd_close` KHÔNG NaN trên spine.

    Chỉ ép `usdvnd_close` (mức tỷ giá đã ffill ở Bước 3, leading_nan=0); các cột
    FX khác (open/high/low/volume) có thể NaN trên phiên ffill — không sao, chỉ
    `close` mới vào feature L3.
    """
    must = [*_OHLCV] + [f"vnindex_{c}" for c in _OHLCV] + ["usdvnd_close"]
    bad = [c for c in must if panel[c].isna().any()]
    if bad:
        raise ValueError(f"Cột daily bắt buộc còn NaN (đáng lẽ đầy đủ trên spine): {bad}")


def check_asof_no_leakage(panel: pd.DataFrame) -> None:
    """3. Anti-leakage: mọi dòng có giá trị slow đều thỏa `release_date <= date`."""
    for p in _SLOW_PREFIXES:
        rd = panel[f"{p}_release_date"]
        m = rd.notna()
        if (panel.loc[m, "date"] < rd[m]).any():
            raise ValueError(f"[{p}] LEAKAGE: tồn tại phiên t < release_date.")


def check_leading_nan_only(panel: pd.DataFrame) -> None:
    """4. NaN của cột slow chỉ là PREFIX đầu chuỗi (chứng minh ffill, không bfill)."""
    daily = set(_daily_cols()) | {"date", "fx_ffilled"}
    for c in panel.columns:
        if (c in daily
                or c.endswith("_reference_period")
                or c.endswith("_release_date")):
            continue
        m = panel[c].isna().to_numpy()
        if m.any() and (~m).any():
            first_valid = (~m).argmax()
            if m[first_valid:].any():
                raise ValueError(
                    f"[{c}] có NaN GIỮA chuỗi (không phải prefix) -> sai ffill/bfill."
                )


def validate(panel: pd.DataFrame, spine: pd.DataFrame) -> None:
    """Chạy cả 4 check; fail thì raise ValueError."""
    check_spine_aligned(panel, spine)
    check_daily_complete(panel)
    check_asof_no_leakage(panel)
    check_leading_nan_only(panel)