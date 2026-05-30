# Step 5 / Bước 5 (Phase 1) — Biến đổi & Kỹ thuật đặc trưng (Feature Engineering)

Tài liệu này mô tả Bước 5 đã làm gì và giải thích mã nguồn. Bước 5 hiện thực khâu
**"biến đổi"** trong pipeline Step 2 của `data.md`: biến `integrated.parquet`
(panel raw đã đồng trục, Bước 4) thành **20 feature nhân quả** theo đúng
`research_design.md` §4. **Không** gán nhãn (Bước 6), **không** Z-score (Bước 8).
Đây là **chốt chặn leakage thứ 1/4** (lag `≤ t-1`).

---

## 1. Mục tiêu, Input, Output

| | Nội dung |
| :--- | :--- |
| **Mục tiêu** | Tính 20 feature pre-registered; mọi feature tại `t` chỉ dùng dữ liệu đến hết `t-1` |
| **Input** | `data/interim/integrated.parquet` (Bước 4) + 3 nguồn chậm raw `data/raw/{cpi,gdp,tcb_fundamentals}.parquet` (cho YoY dùng history 2017) |
| **Output** | `data/interim/features_raw.parquet` + `data/interim/_features_log.json` |
| **Lệnh chạy** | `python scripts/features_phase1.py` (exit 0 nếu OK, 1 nếu lỗi) |
| **Kết quả** (29/05/2026) | 1994 phiên × 20 feature, `2018-06-04 → 2026-05-28`, vùng khả dụng từ **2019-06-07**, 4/4 check pass |

**Nguyên tắc thiết kế** (nối tiếp Bước 1–4):
- *Kiến trúc lai*: logic thuần (`features.py`) + runner mỏng (`features_phase1.py`).
- *Fidelity pre-registration*: implement **explicit** từng công thức như `research_design.md` viết (không global shift) → verify leakage được từng feature.
- *Thuần biến đổi*: ra feature thô, chưa nhãn, chưa chuẩn hóa. Giữ leading NaN warmup (cắt ở Bước 7).

---

## 2. Kết quả thật (29/05/2026) — 20 feature

`date` + 20 feature, **1994 phiên**. Một bộ feature dùng **chung** cho cả 4 model
`k∈{1,5,10,20}` — feature nhìn *quá khứ*, chỉ nhãn (Bước 6) mới phân biệt theo `k`.

| Lớp | Feature | Công thức (đã encode `≤ t-1`) |
| :--- | :--- | :--- |
| **L1** (4) | `r1` | $\log(P_{t-1}/P_{t-2})$ |
| | `r5` `r10` `r20` | $\log(P_{t-1}/P_{t-1-h})$, $h=5,10,20$ |
| **L2** (6) | `ma5_20` | $\mathrm{MA}_5/\mathrm{MA}_{20}$ (SMA, qua `t-1`) |
| | `momentum_3_12` | $\log(P_{t-63}/P_{t-252})$ (skip 63 phiên) |
| | `bb_position` | $(P_{t-1}-\mathrm{MA}_{20})/(2\sigma_{20})$ (σ population) |
| | `trb` | $\mathbb 1[P_{t-1}>\max_{t-21..t-2}]-\mathbb 1[P_{t-1}<\min_{t-21..t-2}]\in\{-1,0,1\}$ |
| | `rsi_14` | Wilder RSI(14) qua `t-1` |
| | `macd_norm` | $(\mathrm{EMA}_{12}-\mathrm{EMA}_{26})/P_{t-1}$ (không signal line) |
| **L3** (4) | `vnindex_ret` | $\log(I_{t-1}/I_{t-2})$ |
| | `fx_logchg` | $\log(X_{t-1}/X_{t-2})$ |
| | `cpi_yoy` | $\mathrm{CPI}_t/\mathrm{CPI}_{t-12}-1$ (chuỗi tháng raw) |
| | `gdp_yoy` | $(\mathrm{GDP}_q-\mathrm{GDP}_{q-4})/\mathrm{GDP}_{q-4}$ (chuỗi quý raw) |
| **L4** (6) | `total_assets_yoy` | $\mathrm{TA}_q/\mathrm{TA}_{q-4}-1$ |
| | `pe_ratio` | P/E quý lấy thẳng từ bảng `ratio` provider, `.shift(1)` (as-of như siblings L4) |
| | `npl_ratio` | `npl_ratio_pct` (trực tiếp) |
| | `credit_yoy` | $\mathrm{Credit}_q/\mathrm{Credit}_{q-4}-1$ |
| | `nim` | `nim_pct` (trực tiếp) |
| | `equity_to_assets` | `equity`/`total_assets` |

**Leading-NaN & vùng khả dụng** (cắt ở Bước 7):

| Feature | NaN | Lý do |
| :--- | :--- | :--- |
| `momentum_3_12` | **252** | cần $P_{t-252}$ → **ràng buộc chặt nhất**, định mốc vùng khả dụng |
| `total_assets_yoy`, `credit_yoy` | **236** | YoY quý cần 4 kỳ; raw fundamentals chỉ từ 2018-Q1 (TCB niêm yết 2018) → trần data |
| `npl_ratio`, `nim`, `pe_ratio` | **52** | 51 phiên prefix từ Bước 4 (2018-Q1 thiếu NPL/NIM/P/E) + 1 do `.shift(1)` |
| `r20`, `trb` | 21 | mẫu số/cửa sổ 20 phiên |
| `ma5_20`, `bb_position` | 20 | cửa sổ 20 phiên |
| `rsi_14` | 15 | Wilder warmup 14 + diff |
| `r10`,`r5`,`r1`,`vnindex_ret`,`fx_logchg` | 11/6/2/2/2 | mẫu số tương ứng |
| `cpi_yoy`, `gdp_yoy`, `equity_to_assets`, `macd_norm` | 1 | chỉ `.shift(1)` |

→ **`usable_start = 2019-06-07`** (= phiên đầu tiên mọi feature non-NaN, do momentum 252).
Khớp `research_design.md` "vùng dùng được bắt đầu ~giữa 2019".

> **YoY dùng chuỗi kỳ RAW (không phải panel).** Panel sau as-of (Bước 4) chỉ chứa
> các kỳ từ sau spine (sớm nhất 2018-04), mất mẫu số 2017. Nếu tính YoY trên panel,
> `cpi_yoy`/`gdp_yoy` bị warm-up lại 12mo oan (→ 229/227 phiên NaN). Dựng chuỗi kỳ từ
> **nguồn raw** (phủ 2017 — đúng lý do `WARMUP_START=2017-01-01` ở Bước 1) đưa về **1
> phiên**. Đóng đúng vòng lặp thiết kế; deliverable sau warmup không đổi (cả hai cách
> trùng giá trị với mọi kỳ ≥ 2019-04, mà vùng < 2019-06 bị cắt warmup).

---

## 3. Chống leakage `≤ t-1` — hai họ feature (chốt chặn 1/4)

Mọi feature tại `t` chỉ là hàm của dữ liệu đến hết `t-1`; không bao giờ chạm `P_t`
(mốc khởi đầu cửa sổ dự đoán `[t, t+k]`).

**Họ giá-lịch-sử** (L1, L2, `vnindex_ret`, `fx_logchg`): thuần hàm của giá/chỉ
số/tỷ giá quá khứ. Công thức tự encode `t-1` (vd `r1`), hoặc tính chỉ báo rồi
`.shift(1)` (MA, RSI, BB, MACD) — `t-1` là mốc mới nhất được dùng.

**Họ as-of** (`cpi_yoy`, `gdp_yoy`, toàn bộ L4): YoY tính **trên chuỗi kỳ raw** (ghép
kỳ `q` với `q-12`/`q-4` — chính xác, không xấp xỉ từ daily đã ffill), map về daily qua
`*_reference_period` (đã as-of `release_date ≤ t` ở Bước 4) rồi `.shift(1)` cho đồng
nhất `≤ t-1`. Đây là chỗ `reference_period` cố ý giữ ở Bước 4 phát huy tác dụng.

---

## 4. Quy ước khóa (research_design không ghi rõ → chốt ở code)

- **MA** = SMA (rolling mean). **Bollinger** dùng **population std** (`ddof=0`).
- **RSI(14)** = **Wilder smoothing** (`ewm(alpha=1/14, adjust=False)`) — chuẩn Wilder (1978).
- **MACD** = $(\mathrm{EMA}_{12}-\mathrm{EMA}_{26})/P_{t-1}$, **không** signal line (theo research_design; chia $P_{t-1}$ để stationary).
- **P/E**: lấy thẳng từ bảng `ratio` provider (P/E quý), as-of theo `release_date` + `.shift(1)` như siblings L4. (Đổi 30/05/2026: VCI trả `eps_basic_vnd=0` mọi quý nên công thức cũ `P_{t-1}/eps_ttm` bất khả thi; ngoài ra `P_daily/eps_const` ≈ mức giá thô → price-level leak.) NPL/NIM lấy thẳng.

---

## 5. Giải thích mã nguồn

```
scripts/features_phase1.py        ← runner: đọc integrated + 3 raw → build → validate → ghi
└── src/data/features.py          ← logic thuần
    ├── _period_yoy(...)          ← YoY trên chuỗi kỳ RAW đầy đủ → map daily (§2 ghi chú)
    ├── build_features(panel, cpi, gdp, fund)  ← dựng 20 cột explicit theo research_design
    └── validate(...)             ← 4 check
```

- **`features.py`** thuần DataFrame. `FEATURES` (= `L1+L2+L3+L4`) cố định thứ tự cột.
  Họ giá tính trực tiếp + `.shift(1)`; họ as-of qua `_period_yoy(raw, panel, ...)`.
- **`features_phase1.py`** đọc panel + 3 nguồn raw, ghi `features_raw.parquet` và
  `_features_log.json` (kèm `leading_nan` từng feature + `usable_start` để Bước 7 cắt).

---

## 6. Bốn check (hợp đồng chất lượng output)

| Check | Khẳng định |
| :--- | :--- |
| `spine_aligned` | `date` trùng panel: tăng nghiêm ngặt, không trùng, đúng 1994 phiên |
| `feature_set` | Đúng **20 feature**, đúng tên + thứ tự (pre-registered, không thêm/bớt) |
| `no_lookahead` | **Chốt chặn 1/4**: nhiễu hàng cuối panel → không feature họ-giá/L4-trực-tiếp nào đổi. Họ YoY đảm bảo `≤t-1` bằng as-of (Bước 4) + `.shift(1)` tường minh |
| `leading_nan_only` | NaN mỗi feature **chỉ là prefix** warmup, không hố giữa chuỗi |

Lần chạy 29/05/2026: cả 4 pass.

---

## 7. Lưu ý & việc còn nợ

- **Output là interim**, chưa phải `features.parquet`: thiếu **nhãn** (Bước 6) và còn
  **leading NaN warmup** (cắt ở Bước 7). Chưa Z-score (Bước 8).
- **`data.md` §2 "Kết quả"** vẫn cần đồng bộ số fetch (cpi 109, gdp 37, fund 33) trước
  khi lock v1.0 — món nợ từ Bước 1.

**Tiếp theo — Bước 6 (Gán nhãn):** $y_{t,k}=\mathrm{sign}(P_{t+k}-P_t)$, $k\in\{1,5,10,20\}$
— bước **duy nhất** được nhìn tương lai. Tie ($P_{t+k}=P_t$) → `+1` (chốt EDA, tie-rate
`k=1`≈7.8%); `k` phiên cuối mỗi horizon → nhãn NaN (giữ dòng cho inference live, không
vào train).