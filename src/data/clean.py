"""Step 3 (Phase 1) — Làm sạch dữ liệu + dựng trục lịch HOSE (spine).

Logic thuần: DataFrame vào → DataFrame ra, KHÔNG đụng disk (phần I/O nằm ở
``scripts/clean_phase1.py``). Tách lớp như vậy để: (1) unit-test dễ, (2) gọi
trực tiếp nối chuỗi trong một process, (3) Phase 2 chỉ việc bọc script mỏng
quanh các hàm này.

Ranh giới của Bước 3 — CHỈ làm sạch + căn spine. KHÔNG merge (Bước 4),
KHÔNG feature (Bước 5), KHÔNG nhãn (Bước 6), KHÔNG cắt warmup (Bước 7),
KHÔNG Z-score (Bước 8).

Quy tắc khóa (chốt từ EDA Phase 0):
- Spine = tập ngày giao dịch thực của ``tcb_price`` (lịch HOSE). KHÔNG tự sinh
  business-day (sẽ lệch ngày nghỉ lễ VN). ``tcb_price`` là nguồn chân lý.
- TCB & VNINDEX: ngày tăng nghiêm ngặt, ``close > 0``, KHÔNG ffill giá/return.
  VNINDEX thiếu trên một phiên spine => LỖI CỨNG (chỉ số luôn được tính khi sàn
  mở => thiếu là lỗi dữ liệu, không phải nghỉ lễ).
- FX (USD/VND): reindex về spine + ffill MỨC tỷ giá theo as-of backward (lấy
  quan sát gần nhất ≤ phiên). KHÔNG bao giờ bfill — leading NaN (trước quan sát
  FX đầu tiên) để nguyên, sẽ bị cắt ở warmup (Bước 7). Cờ ``fx_ffilled`` đánh
  dấu phiên HOSE mà FX không có quan sát gốc đúng ngày đó.
"""
from __future__ import annotations

import pandas as pd

# Thứ tự cột OHLCV chuẩn (đồng bộ với src/data/fetch_prices.py)
OHLCV_ORDER = ["date", "open", "high", "low", "close", "volume", "fetched_at"]


# ----------------------------------------------------------------------------
# Tiện ích nội bộ
# ----------------------------------------------------------------------------
def _normalize_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Chuẩn hoá cột ngày về datetime naive (drop tz), midnight; sort + dedup."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.tz_localize(None).dt.normalize()
    d = (
        d.sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
        .reset_index(drop=True)
    )
    return d


def _order_cols(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    cols = [c for c in OHLCV_ORDER if c in df.columns] + (extra or [])
    return df[cols]


def _assert_positive_close(df: pd.DataFrame, name: str) -> None:
    """close phải > 0 (bỏ qua NaN — leading NaN của FX xử lý riêng)."""
    p = pd.to_numeric(df["close"], errors="coerce")
    n_bad = int((p <= 0).sum())
    if n_bad > 0:
        raise ValueError(f"[{name}] có {n_bad} phiên close <= 0")


# ----------------------------------------------------------------------------
# 1) TCB — nguồn chân lý của spine
# ----------------------------------------------------------------------------
def clean_tcb_price(tcb_raw: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch giá TCB: sort + dedup ngày, assert close > 0. KHÔNG ffill."""
    d = _normalize_dates(tcb_raw)
    _assert_positive_close(d, "tcb_price")
    return _order_cols(d)


def build_spine(tcb_clean: pd.DataFrame) -> pd.DatetimeIndex:
    """Spine = lịch HOSE = ngày của TCB đã sạch (tăng nghiêm ngặt, không trùng)."""
    spine = pd.DatetimeIndex(tcb_clean["date"], name="date").sort_values()
    if spine.has_duplicates:
        dups = spine[spine.duplicated()].unique()
        raise ValueError(f"[spine] có ngày trùng: {[d.date().isoformat() for d in dups[:10]]}")
    if not spine.is_monotonic_increasing:
        raise ValueError("[spine] ngày không tăng nghiêm ngặt sau khi sort")
    return spine


# ----------------------------------------------------------------------------
# 2) VNINDEX — căn về spine, KHÔNG ffill
# ----------------------------------------------------------------------------
def align_vnindex(vnindex_raw: pd.DataFrame, spine: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex VNINDEX về spine (KHÔNG method). Phiên spine nào VNINDEX thiếu
    => raise (lỗi dữ liệu, không ffill chỉ số)."""
    d = _normalize_dates(vnindex_raw).set_index("date")
    aligned = d.reindex(spine)  # không method => ngày thiếu thành NaN
    missing = aligned.index[aligned["close"].isna()]
    if len(missing) > 0:
        head = [x.date().isoformat() for x in missing[:10]]
        raise ValueError(
            f"[vnindex] thiếu {len(missing)} phiên spine (HOSE mở mà VNINDEX trống) — "
            f"lỗi dữ liệu, KHÔNG ffill chỉ số. Vài ngày đầu: {head}"
        )
    out = aligned.reset_index()
    _assert_positive_close(out, "vnindex")
    return _order_cols(out)


# ----------------------------------------------------------------------------
# 3) FX (USD/VND) — reindex về spine + ffill mức tỷ giá (as-of backward)
# ----------------------------------------------------------------------------
def align_fx(usdvnd_raw: pd.DataFrame, spine: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex FX về spine + ffill MỨC tỷ giá. method='ffill' lấy quan sát gần
    nhất ≤ phiên TỪ CHUỖI GỐC FX (kể cả ngày không thuộc spine) => as-of đúng.
    KHÔNG bfill: leading NaN để nguyên. Thêm cờ ``fx_ffilled``."""
    d = _normalize_dates(usdvnd_raw)
    fx_dates = set(d["date"])
    aligned = d.set_index("date").reindex(spine, method="ffill")
    out = aligned.reset_index()
    # ffilled = phiên spine KHÔNG có quan sát FX gốc đúng ngày & đã được lấp giá trị
    out["fx_ffilled"] = (~out["date"].isin(fx_dates)) & out["close"].notna()
    return _order_cols(out, extra=["fx_ffilled"])