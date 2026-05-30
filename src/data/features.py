"""
src/data/features.py — Bước 5 (Phase 1, Step 2): Biến đổi & Kỹ thuật đặc trưng.

Logic THUẦN (DataFrame in -> DataFrame out), không I/O. Runner mỏng ở
`scripts/features_phase1.py` lo đọc/ghi file.

Biến `integrated.parquet` (panel raw đã đồng trục, Bước 4) + 3 nguồn chậm raw
(`cpi/gdp/fund`, phủ về 2017 cho CPI/GDP) thành 20 feature nhân quả theo đúng
`research_design.md` §4. KHÔNG gán nhãn (Bước 6), KHÔNG Z-score (Bước 8).

CHỐT CHẶN LEAKAGE 1/4: mọi feature tại t chỉ là hàm của dữ liệu đến hết t-1
(không bao giờ chạm P_t — mốc khởi đầu cửa sổ dự đoán). Hai họ:
  - Họ giá-lịch-sử (L1, L2, VNI/FX của L3): hàm của giá/chỉ số/tỷ giá quá khứ;
    công thức tự encode t-1 (vd r1 = log(P_{t-1}/P_{t-2})).
  - Họ as-of (CPI/GDP YoY của L3, toàn bộ L4): YoY tính TRÊN CHUỖI KỲ RAW ĐẦY ĐỦ
    (dùng nguồn raw có history 2017 -> đủ mẫu số q-12/q-4 ngay từ đầu vùng spine),
    map về daily qua *_reference_period (đã as-of release_date<=t ở Bước 4), rồi
    .shift(1) cho đồng nhất "<= t-1" với họ giá.

Implement EXPLICIT từng công thức pre-registered (không global shift) -> fidelity
tối đa + verify leakage được (xem check_no_lookahead).

Quy ước khóa (research_design không ghi rõ -> chốt ở đây):
  - MA = SMA (rolling mean). Bollinger dùng population std (ddof=0).
  - RSI(14) = Wilder smoothing (ewm alpha=1/14) — chuẩn Wilder (1978).
  - MACD = (EMA12 - EMA26)/P_{t-1}, KHÔNG signal line (theo research_design).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 20 feature theo 4 lớp (thứ tự cố định = thứ tự cột output).
L1 = ["r1", "r5", "r10", "r20"]
L2 = ["ma5_20", "momentum_3_12", "bb_position", "trb", "rsi_14", "macd_norm"]
L3 = ["vnindex_ret", "fx_logchg", "cpi_yoy", "gdp_yoy"]
L4 = ["total_assets_yoy", "pe_ratio", "npl_ratio", "credit_yoy", "nim", "equity_to_assets"]
FEATURES = L1 + L2 + L3 + L4  # 20


def _period_yoy(raw: pd.DataFrame, panel: pd.DataFrame, value_col: str,
                periods: int, panel_period_col: str) -> pd.Series:
    """YoY/growth trên CHUỖI KỲ RAW ĐẦY ĐỦ -> map về daily (as-of t, CHƯA lag t-1).

    Dùng nguồn raw (cpi/gdp/fund) — phủ về 2017 cho CPI/GDP — để có đủ mẫu số YoY
    (kỳ q-`periods`) NGAY từ đầu vùng spine; KHÔNG dựng chuỗi kỳ từ panel (panel đã
    bị cắt còn các kỳ sau spine -> mất mẫu số 2017 -> YoY phải re-warm 12mo/4q oan).
    Map ngược về từng phiên qua `panel[panel_period_col]` (đã as-of release_date<=t
    ở Bước 4). periods = 12 (CPI tháng) / 4 (GDP, fundamentals quý).

    Leakage an toàn: kỳ active tại t có release<=t; mẫu số q-`periods` release còn
    sớm hơn -> đều <= t. Ràng buộc <= t-1 do caller .shift(1).
    """
    s = (raw[["reference_period", value_col]]
         .dropna(subset=["reference_period", value_col])
         .drop_duplicates(subset=["reference_period"])
         .set_index("reference_period")[value_col]
         .sort_index())
    yoy = s / s.shift(periods) - 1.0
    return panel[panel_period_col].map(yoy)


def build_features(panel: pd.DataFrame, cpi: pd.DataFrame, gdp: pd.DataFrame,
                   fund: pd.DataFrame) -> pd.DataFrame:
    """Panel raw (Bước 4) + 3 nguồn chậm raw -> `date` + 20 feature thô (giữ leading NaN)."""
    panel = panel.sort_values("date").reset_index(drop=True)
    P = panel["close"]            # giá đóng cửa điều chỉnh TCB
    I = panel["vnindex_close"]    # chỉ số VN-Index
    X = panel["usdvnd_close"]     # tỷ giá USD/VND (mức đã ffill ở Bước 3)

    f = pd.DataFrame({"date": panel["date"]})

    # ---- L1: log-return tích lũy, mốc mới nhất P_{t-1} -------------------------
    f["r1"]  = np.log(P.shift(1) / P.shift(2))
    f["r5"]  = np.log(P.shift(1) / P.shift(6))
    f["r10"] = np.log(P.shift(1) / P.shift(11))
    f["r20"] = np.log(P.shift(1) / P.shift(21))

    # ---- L2: kỹ thuật, tất cả 'as of t-1' -------------------------------------
    ma5  = P.rolling(5).mean().shift(1)
    ma20 = P.rolling(20).mean().shift(1)
    f["ma5_20"] = ma5 / ma20

    f["momentum_3_12"] = np.log(P.shift(63) / P.shift(252))  # skip 63 phiên gần nhất

    sd20 = P.rolling(20).std(ddof=0).shift(1)                # population std
    f["bb_position"] = (P.shift(1) - ma20) / (2.0 * sd20)

    roll_max = P.rolling(20).max().shift(2)  # max trên [t-21, t-2]
    roll_min = P.rolling(20).min().shift(2)  # min trên [t-21, t-2]
    trb = (P.shift(1) > roll_max).astype("float64") - (P.shift(1) < roll_min).astype("float64")
    trb[roll_max.isna() | roll_min.isna() | P.shift(1).isna()] = np.nan  # giữ leading NaN
    f["trb"] = trb

    delta = P.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    f["rsi_14"] = (100.0 - 100.0 / (1.0 + rs)).shift(1)

    macd_line = P.ewm(span=12, adjust=False).mean() - P.ewm(span=26, adjust=False).mean()
    f["macd_norm"] = macd_line.shift(1) / P.shift(1)

    # ---- L3: 2 daily kiểu giá + 2 YoY chuỗi kỳ raw ----------------------------
    f["vnindex_ret"] = np.log(I.shift(1) / I.shift(2))
    f["fx_logchg"]   = np.log(X.shift(1) / X.shift(2))
    f["cpi_yoy"] = _period_yoy(cpi, panel, "cpi_index", 12, "cpi_reference_period").shift(1)
    f["gdp_yoy"] = _period_yoy(gdp, panel, "nominal_gdp_vnd_bil", 4, "gdp_reference_period").shift(1)

    # ---- L4: cơ bản TCB (mix daily x quarterly), tất cả '<= t-1' ---------------
    f["total_assets_yoy"] = _period_yoy(fund, panel, "total_assets_vnd_bil", 4, "fund_reference_period").shift(1)
    f["credit_yoy"]       = _period_yoy(fund, panel, "credit_balance_vnd_bil", 4, "fund_reference_period").shift(1)
    f["pe_ratio"]         = panel["pe_ratio"].shift(1)
    f["npl_ratio"]        = panel["npl_ratio_pct"].shift(1)
    f["nim"]              = panel["nim_pct"].shift(1)
    f["equity_to_assets"] = (panel["equity_vnd_bil"] / panel["total_assets_vnd_bil"]).shift(1)

    return f[["date", *FEATURES]]


# =============================================================================
# Kiểm định — 4 check (gương Bước 3–4)
# =============================================================================
def check_spine_aligned(features: pd.DataFrame, panel: pd.DataFrame) -> None:
    """1. `date` trùng panel: tăng nghiêm ngặt, không trùng, đúng tập."""
    d = features["date"]
    if not d.is_monotonic_increasing or d.duplicated().any():
        raise ValueError("features.date không tăng nghiêm ngặt / có trùng.")
    p = panel.sort_values("date")["date"].reset_index(drop=True)
    if not d.reset_index(drop=True).equals(p):
        raise ValueError("features.date lệch panel.")


def check_feature_set(features: pd.DataFrame) -> None:
    """2. Đúng 20 feature, đúng tên, đúng thứ tự (pre-registered, không thêm/bớt)."""
    expected = ["date", *FEATURES]
    if list(features.columns) != expected:
        raise ValueError(f"Cột feature sai.\n expected={expected}\n got={list(features.columns)}")


def check_no_lookahead(panel: pd.DataFrame, cpi: pd.DataFrame, gdp: pd.DataFrame,
                       fund: pd.DataFrame) -> None:
    """3. LEAKAGE 1/4 (họ giá + L4 trực tiếp): nhiễu HÀNG CUỐI panel raw -> KHÔNG đổi.

    Mọi feature tại t chỉ dùng <= t-1, nên dữ liệu hàng cuối (T) không bao giờ vào
    feature nào (chỉ có thể vào hàng T+1 — không tồn tại). Nếu một feature lỡ dùng
    giá trị hàng-của-chính-nó/tương lai, nhiễu hàng cuối sẽ làm đổi feature -> bắt.

    Phạm vi: check này nhiễu các cột trong PANEL nên phủ họ giá (L1/L2/L3-daily) và
    L4 trực tiếp (pe/npl/nim/equity_to_assets). Họ YoY (cpi/gdp/total_assets/credit)
    lấy từ nguồn raw + map theo reference_period (đã as-of release<=t, Bước 4 đã
    kiểm) rồi .shift(1) tường minh -> ràng buộc <=t-1 đảm bảo bằng 2 cơ chế đó.
    """
    panel = panel.sort_values("date").reset_index(drop=True)
    base = build_features(panel, cpi, gdp, fund)
    p2 = panel.copy()
    last = p2.index[-1]
    for col in ("close", "vnindex_close", "usdvnd_close", "total_assets_vnd_bil",
                "equity_vnd_bil", "pe_ratio", "npl_ratio_pct", "nim_pct"):
        if col in p2.columns and pd.notna(p2.at[last, col]):
            p2.at[last, col] = p2.at[last, col] * 1.5 + 1.0
    pert = build_features(p2, cpi, gdp, fund)

    a, b = base[FEATURES].to_numpy(), pert[FEATURES].to_numpy()
    diff = ~((a == b) | (pd.isna(a) & pd.isna(b)))
    if diff.any():
        bad = [FEATURES[j] for j in range(len(FEATURES)) if diff[:, j].any()]
        raise ValueError(f"LOOKAHEAD: nhiễu hàng cuối làm đổi feature {bad} -> có leak.")


def check_leading_nan_only(features: pd.DataFrame) -> None:
    """4. NaN của mỗi feature CHỈ là prefix đầu chuỗi (warmup), không có hố giữa."""
    for c in FEATURES:
        m = features[c].isna().to_numpy()
        if m.any() and (~m).any():
            first_valid = (~m).argmax()
            if m[first_valid:].any():
                raise ValueError(f"[{c}] có NaN GIỮA chuỗi (không phải prefix warmup).")


def validate(features: pd.DataFrame, panel: pd.DataFrame, cpi: pd.DataFrame,
             gdp: pd.DataFrame, fund: pd.DataFrame) -> None:
    """Chạy cả 4 check; fail thì raise ValueError."""
    check_spine_aligned(features, panel)
    check_feature_set(features)
    check_no_lookahead(panel, cpi, gdp, fund)
    check_leading_nan_only(features)