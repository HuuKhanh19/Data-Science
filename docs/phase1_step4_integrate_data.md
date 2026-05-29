# Step 4 / Bước 4 (Phase 1) — Tích hợp dữ liệu (Data Integration)

Tài liệu này mô tả Bước 4 đã làm gì và giải thích mã nguồn. Bước 4 hiện thực khâu
**"tích hợp"** trong pipeline Step 2 của `data.md`: gộp 6 nguồn (sau Bước 3) thành
**một panel daily** trên trục spine HOSE, mỗi dòng `t` phản ánh đúng *trạng thái
thông tin đã thực sự công bố tính đến hết ngày t*. Đầu ra vẫn là **dữ liệu thô**
(chưa YoY/log-return — để dành Bước 5). Đây là **chốt chặn leakage thứ 2/4**.

---

## 1. Mục tiêu, Input, Output

| | Nội dung |
| :--- | :--- |
| **Mục tiêu** | Đồng trục 6 nguồn về 1 panel daily; với nguồn chậm chỉ cho thấy số đã công bố ≤ `t` (anti-leakage) |
| **Input** | 3 file daily `data/interim/*_clean.parquet` (đã trên spine, Bước 3) + 3 nguồn chậm `data/raw/{cpi,gdp,tcb_fundamentals}.parquet` |
| **Output** | `data/interim/integrated.parquet` + `data/interim/_integrate_log.json` |
| **Lệnh chạy** | `python scripts/integrate_phase1.py` (exit 0 nếu OK, 1 nếu lỗi) |
| **Kết quả** (29/05/2026) | 1994 phiên × 32 cột, `2018-06-04 → 2026-05-28`, 4/4 check pass |

**Nguyên tắc thiết kế** (nối tiếp Bước 1–3):
- *Kiến trúc lai*: logic thuần (`integrate.py`, DataFrame in→out) + runner mỏng (`integrate_phase1.py`, file-in/file-out) → Phase 2 dễ tự động hóa, mỗi bước resume/audit được.
- *Anti-leakage*: nguồn chậm join theo `release_date` (ngày công bố thực), **không** theo `reference_period` (kỳ tham chiếu).
- *Thuần tích hợp*: giữ tên cột thô, không transform; YoY/log-return để Bước 5.
- *Schema lock đầu ra*: 4 check là hợp đồng chất lượng của panel; sai là raise.

---

## 2. Kết quả thật (29/05/2026)

Panel **1994 phiên** (= spine HOSE), `2018-06-04 → 2026-05-28`, **32 cột**:

| Nhóm | Cột |
| :--- | :--- |
| Trục | `date` |
| TCB (subject, tên trần) | `open, high, low, close, volume` |
| VNINDEX (prefix) | `vnindex_{open,high,low,close,volume}` |
| USD/VND (prefix) + cờ | `usdvnd_{open,high,low,close,volume}`, `fx_ffilled` |
| CPI (L3) | `cpi_reference_period`, `cpi_index`, `cpi_release_date` |
| GDP (L3) | `gdp_reference_period`, `nominal_gdp_vnd_bil`, `gdp_release_date` |
| Fundamentals (L4) | `fund_reference_period`, `fund_release_date`, `total_assets_vnd_bil`, `equity_vnd_bil`, `net_interest_income_vnd_bil`, `npl_ratio_pct`, `credit_balance_vnd_bil`, `eps_ttm`, `nim_pct` |

**Leading-NaN (cột nguồn chậm):** tất cả = 0, **riêng `npl_ratio_pct` & `nim_pct` = 51**.
Giải thích: từ phiên đầu spine (2018-06-04), bản fundamentals active là **2018-Q1**
— quý đầu tiên TCB có dữ liệu, nhưng NPL/NIM thiếu tại nguồn và không có quý trước
để ffill. Prefix kéo dài tới khi **2018-Q2** release (`2018-06-30 + 45 = 2018-08-14`),
đúng 51 phiên HOSE trong `2018-06-04 → 2018-08-13`. Đây là **leading NaN hợp lệ**,
sẽ bị cắt tự nhiên ở **Bước 7** (warmup tới ~giữa 2019), không phải lỗi.

> Lưu ý: `interest_earning_assets_vnd_bil` (33/33 NaN, vnstock không cung cấp) đã bị
> **bỏ** ở tích hợp theo chốt EDA Phase 0 — nên L4 có 7 cột giá trị, không phải 8.
> Các cột FX ngoài `close` (open/high/low/volume) không vào feature L3; chỉ
> `usdvnd_close` là "mức tỷ giá" được dùng.

---

## 3. Hai cơ chế ghép — khác bản chất

### 3.1 Daily Merge (VNINDEX, USD/VND) — `merge_daily`
Ba nguồn daily đã **cùng trục spine** ở Bước 3, nên ghép chỉ là **gán cột theo
`date`**: hàm yêu cầu tập ngày của VNINDEX/USD-VND **trùng khớp tuyệt đối** spine,
lệch là lỗi data → raise (không bao giờ tự sinh/điền ngày). Không lệch tần suất,
không rủi ro nhìn tương lai: phiên `t` là dữ liệu của chính phiên `t` (lag về `t-1`
để thành feature là việc Bước 5). TCB OHLCV giữ **tên trần** (subject, `P_t=close`);
VNINDEX/USD-VND **prefix** để tránh đụng cột cùng tên.

### 3.2 As-of Join backward (CPI/GDP/fundamentals) — `asof_join`
Đây là lý do Bước 4 tồn tại. Macro/fundamental tần suất thấp **và trễ release**:
CPI tháng 3 không tồn tại vào 31/3 mà công bố quãng 6/4. Nếu join theo
`reference_period`, tại phiên 1/4 mô hình đã "thấy" CPI tháng 3 → **leakage**.

`pd.merge_asof(direction="backward")` với `left_on="date"`, `right_on="release_date"`
chữa đúng chỗ đó: tại mỗi phiên `t` lấy bản ghi có **`release_date ≤ t` và gần `t`
nhất** — "thông tin mới nhất đã thực sự công bố tính đến hết ngày t". Quy ước
`release_date` bảo thủ chốt ở Bước 1 (CPI +6, GDP +30, fund +45 ngày) chính là tham
số khóa của bước này. Giữ lại `*_reference_period`/`*_release_date` (đổi tên có
prefix) để **audit** và để **Bước 5** dựng lại chuỗi kỳ tính YoY trên chuỗi raw.

---

## 4. Hai loại forward-fill — đừng nhầm

| Loại | Ở đâu | Làm gì | An toàn leakage vì |
| :--- | :--- | :--- | :--- |
| ffill **giữa chuỗi kỳ** | `prep_slow`, trên chuỗi quý/tháng | Lấp field thiếu *trong* nguồn chậm (fundamentals **2021-Q2** NPL/NIM ← **2021-Q1**) | Q1 công bố (~05/2021) trước cửa sổ Q2 active (~08–11/2021) |
| ffill **giữa release** | `asof_join`, bản chất `direction="backward"` | Kéo bản công bố gần nhất xuống mọi phiên daily tới kỳ kế ("biết gì dùng nấy") | Chỉ kéo số đã có `release_date ≤ t` |

NaN dẫn đầu (2018-Q1 NPL/NIM) **không** lấp được bằng ffill → để nguyên thành prefix
→ cắt ở Bước 7. **Không bao giờ bfill** (bfill = nhìn tương lai).

---

## 5. Giải thích mã nguồn

```
scripts/integrate_phase1.py        ← runner: đọc interim+raw → integrate → validate → ghi
└── src/data/integrate.py          ← logic thuần
    ├── merge_daily(...)           ← §3.1 ghép daily theo date (+ raise nếu lệch spine)
    ├── prep_slow(...)             ← §4 sort theo reference_period, bỏ cột chết, ffill chuỗi kỳ
    ├── asof_join(...)             ← §3.2 merge_asof backward theo release_date
    ├── integrate(...)             ← orchestrate: merge_daily → 3× asof_join
    └── validate(...)              ← §6 chạy 4 check
```

- **`integrate.py`** thuần DataFrame, không I/O — test/tái dùng dễ. Hằng số khóa:
  `_OHLCV`, `_FUND_DROP` (cột chết), `_SLOW_PREFIXES`.
- **`integrate_phase1.py`** đọc 3 interim + 3 raw, gọi `integrate` + `validate`, ghi
  `integrated.parquet` và `_integrate_log.json` (kèm `leading_nan_slow` để verify
  ranh giới warmup). Bắt exception → ghi log lỗi, exit 1 (tiện Phase 2).

---

## 6. Bốn check (hợp đồng chất lượng output)

| Check | Khẳng định |
| :--- | :--- |
| `spine_aligned` | `date` của panel **trùng khớp spine**: tăng nghiêm ngặt, không trùng, đúng tập 1994 phiên |
| `daily_complete` | TCB & VNINDEX OHLCV + `usdvnd_close` **không NaN** trên spine |
| `asof_no_leakage` | **Mọi** dòng có giá trị slow đều thỏa `release_date ≤ date` (chốt chặn 2/4) |
| `leading_nan_only` | NaN cột slow **chỉ là prefix** đầu chuỗi → chứng minh ffill, **không** bfill |

Lần chạy 29/05/2026: cả 4 pass.

---

## 7. Lưu ý & việc còn nợ

- **Đồng bộ `data.md` §2 "Kết quả"** với lần fetch mới (cpi 109, gdp 37, fund 33)
  trước khi lock v1.0 — việc nhỏ còn nợ từ Bước 1, chưa làm.
- **Output là interim**, chưa phải `features.parquet`. Panel này là *raw đã đồng
  trục*; chưa có feature, chưa có nhãn.

**Tiếp theo — Bước 5 (Biến đổi & Feature Engineering):** từ `integrated.parquet`
tính 20 feature nhân quả (L1 log-return, L2 technical, L3 vĩ mô YoY, L4 cơ bản YoY),
mọi feature tại `t` chỉ dùng dữ liệu đến `P_{t-1}` (chốt chặn leakage 1/4 — lag).
YoY L3/L4 dựng lại từ `*_reference_period` để tính trên chuỗi kỳ (chính xác). Hoãn
Z-score sang Bước 8.