# Step 1 (Phase 1) — Thu thập Raw Data

Tài liệu này mô tả những gì Step 1 đã làm được và giải thích chi tiết mã nguồn. Step 1 hiện thực **khâu "thu thập"** trong pipeline data của `data.md`: lấy 6 nguồn raw, kiểm định, lưu thành parquet — *chưa* biến đổi (transform để dành Step 2).

---

## 1. Mục tiêu, Input, Output

| | Nội dung |
| :--- | :--- |
| **Mục tiêu** | Thu thập trung thực 6 nguồn dữ liệu (2018-06-04 → 2026-05-29, **snapshot tĩnh Phase 1**), kiểm định chất lượng, lưu parquet schema-locked |
| **Input** | API/web ngoài: vnstock (VCI), yfinance, IMF Data Portal (SDMX), VBMA GDP TSV |
| **Output** | 6 file `data/raw/*.parquet` + `data/raw/_fetch_log.json` (nhật ký + thống kê mỗi nguồn) |
| **Lệnh chạy** | `python scripts/fetch_phase1.py` |
| **Kết quả** (snapshot 29/05/2026) | 6/6 OK — `tcb_price` 1994, `vnindex` 2086, `usdvnd` 2078, `cpi` 109, `gdp` 37, `tcb_fundamentals` 33 |

**Nguyên tắc thiết kế** (xuyên suốt mọi module):
- *Raw-only*: chỉ thu thập, không tính feature (YoY, log-return... để Step 2).
- *Schema lock*: mỗi output có hợp đồng cột/null/khóa chính, sai là raise ngay.
- *Anti-leakage từ gốc*: biến chậm lưu kèm `release_date` (ngày công bố thực hoặc quy ước bảo thủ).
- *Audit*: mọi bảng có `fetched_at` (giờ VN).
- *Snapshot đông cứng*: Phase 1 là nghiên cứu confirmatory → `END_DATE` **khóa cứng** = `2026-05-29` (không dùng `now()`) để dataset tái lập được. Phase 2 (tự động hóa) mới quay lại fetch động.

---

## 2. Kiến trúc & bản đồ file

Tổ chức theo **một file một nguồn dữ liệu** (đặt theo data raw mỗi file collect được), giúp mỗi cơ chế thu thập và mỗi rủi ro nguồn của nó nằm tách bạch:

```
scripts/fetch_phase1.py          ← orchestrator: gọi 5 nguồn, ghi log + summary
└── src/data/
    ├── schema.py                ← "hợp đồng dữ liệu": 6 ParquetSchema + .validate()
    ├── validation.py            ← validators chất lượng cho dữ liệu giá
    ├── _common.py               ← helper chung: _now_vn, _http_get_bytes,
    │                              _normalize_ohlcv, _save_and_report
    ├── fetch_prices.py          ← tcb_price, vnindex   (vnstock Quote VCI — adjusted close)
    ├── fetch_fx.py              ← usdvnd               (yfinance USDVND=X)
    ├── fetch_cpi.py             ← cpi                  (IMF SDMX, sdmx1)
    ├── fetch_gdp.py             ← gdp                  (VBMA TSV — web scrape)
    └── fetch_fundamentals.py    ← tcb_fundamentals     (vnstock VCIFinance method private)
```

Mỗi hàm `fetch_*` trả về một **dict báo cáo** đồng nhất: `{status, rows, date_min, date_max, output, ...}` để orchestrator tổng hợp.

> **Vì sao tách theo nguồn, không theo tần suất.** Bản cũ gộp theo "channel" daily/tháng/quý: `fetch_prices` chứa cả vnstock (giá) lẫn yfinance (FX), `fetch_macro` gộp một API số sạch (CPI) với một scrape bẩn (GDP). Tách theo nguồn để mỗi file đúng một cơ chế — sửa parser GDP không đụng CPI, đổi nguồn FX không đụng giá.

---

## 3. Giải thích mã nguồn

### 3.1 `scripts/fetch_phase1.py` — Orchestrator

Điều phối toàn bộ. `START_DATE = "2018-06-04"` (ngày niêm yết TCB), `WARMUP_START = "2017-01-01"` (lùi ≥4 quý cho YoY của GDP & fundamentals), `END_DATE = PHASE1_SNAPSHOT = "2026-05-29"` (**khóa cứng** — snapshot tĩnh Phase 1; Phase 2 mới dùng `datetime.now()`).

- **`_run_one(name, fn, **kwargs)`**: wrapper gọi một hàm fetch, in tiến trình, bắt mọi exception và quy về `{status:"error", ...}` để **một nguồn lỗi không làm sập cả pipeline**.
- **`main()`**: tạo `data/raw/`, lần lượt chạy 5 nguồn (giá → FX → CPI → GDP → fundamentals), gom kết quả vào `results`, ghi `_fetch_log.json`, in summary (đếm OK/warning/error). Trả exit code `0` nếu không có error, `1` nếu có (tiện cho tự động hóa ở Phase 2).

Import (sau refactor):
```python
from src.data.fetch_prices import fetch_tcb_price, fetch_vnindex
from src.data.fetch_fx import fetch_usdvnd
from src.data.fetch_cpi import fetch_cpi
from src.data.fetch_gdp import fetch_gdp
from src.data.fetch_fundamentals import fetch_tcb_fundamentals
```

### 3.2 `src/data/schema.py` — Hợp đồng dữ liệu

Định nghĩa cấu trúc bắt buộc của từng output, tách biệt khỏi logic fetch.

- **`ColumnSpec`**: tên cột, dtype, `nullable`, mô tả.
- **`ParquetSchema.validate(df)`** kiểm 3 điều, fail thì raise `ValueError`:
  1. **Đủ cột**: thiếu cột khai báo → lỗi.
  2. **Ràng buộc non-null**: cột `nullable=False` mà có null → lỗi (vd `close`, `reference_period`).
  3. **Khóa chính không trùng**: vd `date` (giá) hoặc `reference_period` (CPI/GDP/fundamentals) bị lặp → lỗi.
- 6 schema cụ thể: `TCB_PRICE / VNINDEX / USDVND` dùng chung khung **OHLCV** (`date, open, high, low, close, volume, fetched_at`, khóa `date`); `CPI` (`cpi_index`, `release_date`), `GDP`  (`nominal_gdp_vnd_bil`, `release_date`), `TCB_FUNDAMENTALS` (10 cột chỉ số + `release_date`).

### 3.3 `src/data/validation.py` — Kiểm định chất lượng giá

Trả về `ValidationReport(warnings, errors)`. Bốn kiểm tra:

- **`check_monotonic_dates`**: ngày phải tăng nghiêm ngặt, không trùng (error nếu vi phạm).
- **`check_hose_calendar_gap`**: gap giữa 2 phiên > 12 ngày → **warning**, > 15 ngày → **error** (cho phép gap Tết ~7–10 ngày là hợp lệ).
- **`check_abnormal_returns`**: |log-return| ngày > 15% → cảnh báo (dấu hiệu adjustment chưa apply hoặc lỗi data); giá ≤ 0 → error. *Trên adjusted close đúng, TCB có 0 phiên vi phạm dù chia cổ tức cổ phiếu 1:1 năm 2024.*
- **`canonicalize_price_unit`**: median giá phải nằm trong khoảng kỳ vọng (TCB: [5, 200] nghìn VND); lệch → raise để **người kiểm tra**, *không tự convert* (an toàn hơn).

### 3.4 `src/data/_common.py` — Helper dùng chung

Gom 4 tiện ích từng bị lặp ở các module fetch về một chỗ (single source of truth):

- **`_now_vn()`**: timestamp giờ VN (`Asia/Ho_Chi_Minh`) cho cột `fetched_at`.
- **`_http_get_bytes(url, retries, timeout, verify_ssl)`**: GET có retry + User-Agent, tùy chọn tắt verify SSL — dùng cho scrape GDP.
- **`_normalize_ohlcv(df, rename_map)`**: chuẩn hoá về khung OHLCV — reset index nếu ngày là index, **drop timezone** (yfinance tz-aware), bỏ cột thừa (`Adj Close, Dividends, Stock Splits`), lowercase tên cột, ép kiểu, **drop dòng `close` null** (yfinance đôi khi trả NaN cho phiên "hôm nay"), sort + drop trùng ngày, gắn `fetched_at`. Dùng chung cho cả vnstock lẫn yfinance.
- **`_save_and_report(df, name, schema, validators)`**: chạy `schema.validate` → chạy các validator → ghi parquet → trả dict báo cáo.

### 3.5 `src/data/fetch_prices.py` — Giá daily (vnstock VCI)

- **`fetch_tcb_price`**: `vnstock.Quote(symbol="TCB", source="VCI").history(...)`. **VCI trả adjusted close** (đã xử lý corporate action). Chạy đủ 3 validator + `canonicalize_price_unit`.
- **`fetch_vnindex`**: tương tự nhưng **bỏ** `canonicalize_price_unit` (VN-Index ~800–1500, khác đơn vị) và bỏ `check_abnormal_returns` (chỉ số ít khi nhảy >15%).

### 3.6 `src/data/fetch_fx.py` — Tỷ giá USD/VND (yfinance)

- **`fetch_usdvnd`**: `yfinance.Ticker("USDVND=X").history(..., auto_adjust=False)`. **Bỏ** `check_hose_calendar_gap` vì FX chạy lịch 24/5 khác HOSE.
- Tách riêng khỏi `fetch_prices` vì cơ chế khác hẳn (yfinance, không phải vnstock); không canonicalize unit (USD/VND ~22.000–27.000, đơn vị thô VND/USD).

### 3.7 `src/data/fetch_cpi.py` — CPI (IMF SDMX)

- **Nguồn**: IMF SDMX, dataset `CPI`, key `VNM.CPI._T.IX.M` (Việt Nam, all-items `_T`, dạng index `IX`, tần suất tháng `M`), gốc 2024=100. Gọi qua `sdmx1`: `sdmx.Client("IMF_DATA").data("CPI", key=..., params={"startPeriod": ...})`.
- **Trích chuỗi**: `sdmx.to_pandas(msg)` → lấy level `TIME_PERIOD` ("2017-M01") map về month-end; cột `cpi_index` ép numeric.
- **release_date**: nguồn số không có ngày công bố → quy ước `= reference_period (month-end) + 6 ngày` (NSO thực tế ra CPI tháng M vào ~mùng 3–6 tháng M+1 → an toàn leakage).
- **Warmup**: `startPeriod = start_date − prehistory_months` (mặc định 15) để đủ ≥12 tháng trước phiên đầu panel cho `cpi_yoy` (tính ở Step 2). Lọc `reference_period ∈ [cutoff, end_date]`, validate, lưu (`status="ok"` nếu ≥ 50 dòng).
- *Vì sao bỏ scrape NSO*: press-release text không đồng nhất (≈49% tháng thiếu MoM/point-YoY) → không đảm bảo full coverage. IMF cho chuỗi số liền mạch; YoY tự tính khớp đúng số chính thức GSO (vd 2026-03: 4.65%).

### 3.8 `src/data/fetch_gdp.py` — GDP (VBMA scrape)

- Tải bytes (qua `_http_get_bytes`), **dò encoding**: file là **UTF-16 LE** (BOM `\xff\xfe`) — code thử lần lượt `utf-16, utf-8-sig, utf-8, cp1252, latin-1` và chỉ chấp nhận encoding nào parse ra **cột dạng "Q\<n\> YYYY"** (chặn latin-1 decode "thành công" nhưng ra ký tự rác).
- File là **TSV** (`sep="\t"`), layout B: cột = nhãn quý, một hàng nhãn "Nominal Gross Domestic Product". Code tìm đúng hàng đó, parse giá trị dạng `"809,613"` (bỏ ngoặc kép + dấu phẩy ngăn nghìn) → list `(quarter_end, value)`.
- Gán `release_date = quarter_end + 30 ngày` (quy ước bảo thủ vì VBMA không cung cấp ngày công bố), lọc theo khoảng, validate, lưu. `verify_ssl=False` mặc định (VBMA lỗi cert).

### 3.9 `src/data/fetch_fundamentals.py` — Cơ bản TCB (vnstock VCIFinance private)

Điểm mấu chốt: API public của vnstock **chỉ trả 4 quý**. Code **gọi thẳng method private** để mở khoá full history.

- **`_fetch_full_history(symbol)`**: import `vnstock.explorer.vci.financial.Finance` (VCIFinance), gọi `_get_financial_report(stmt_type, period="quarter", lang="en", get_all=True, limit=100)` cho `balance_sheet / ratio / income_statement` → mỗi statement ~31–33 quý. Có **fallback** về method public (4 quý) nếu private call lỗi. (`redirect_stdout` để chặn banner quảng cáo vnstock.)
- **`FIELD_MAPPING`**: ánh xạ tên feature → `(statement, [item_id ưu tiên])`, vd `credit_balance → loans_and_advances_to_customers_net`, `npl_ratio_pct → npl`, `nim_pct → net_interest_margin`.
- **`_find_item_row` / `_extract_quarter_series`**: tìm đúng dòng theo `item_id`, trích các cột dạng `YYYY-Qn` thành chuỗi `{quarter_end: value}`.
- **`_normalize_to_billion`**: nếu giá trị quá lớn (median > 1e9, tức đơn vị VND) thì chia 1e9 về **tỷ VND**; bỏ qua với cột `_pct`, `eps` và `pe_ratio` (đều không phải đơn vị tiền).
- **`fetch_tcb_fundamentals`**: ghép tất cả field theo `reference_period`, cột `interest_earning_assets` để `NA` (vnstock không có), gán `release_date = quý + 45 ngày`, sắp đúng thứ tự cột theo schema, validate, lưu. Báo cáo kèm `null_counts` và `field_sources` (truy vết mỗi feature lấy từ item_id nào).

> Module này giữ nguyên (không gộp vào `_common`); vẫn tự định nghĩa `_now_vn` cục bộ — chấp nhận một chút trùng lặp để không đụng code đã chạy ổn.

---

## 4. Lưu ý & giới hạn đã biết

- **Snapshot đông cứng (Phase 1)**: `END_DATE` khóa cứng = `2026-05-29` để confirmatory study tái lập đúng vùng dữ liệu. **Lưu ý khóa tham số ≠ khóa artifact**: adjusted close có thể bị VCI *re-adjust* nếu TCB có corporate action sau ngày freeze (scale lại toàn chuỗi giá), nên snapshot raw cần được **bảo toàn** (commit hoặc copy sang thư mục archive read-only) thay vì chỉ trông vào regenerate. Phase 2 mới quay lại `END_DATE = now()` cho fetch động.
- **`.gitignore` & dọn dẹp**: thư mục `_debug/` và các script chẩn đoán đã được dọn ở bước clean. Với confirmatory design, nên đưa snapshot `data/raw/` Phase 1 vào diện được giữ (commit/archive) thay vì gitignore hoàn toàn — dataset đã "đăng ký" phải sống cùng repo.
- **Phụ thuộc nguồn ngoài** (rủi ro cần theo dõi khi auto-refit ở Phase 2): giá (vnstock Quote) + FX (yfinance) + CPI (IMF SDMX) là API chuẩn, khá bền. Hai điểm dễ vỡ: **GDP scrape VBMA** (đổi layout/encoding → parser raise rõ ràng, cần cập nhật) và **fundamentals gọi method private VCIFinance** (vnstock đổi internal giữa các version có thể làm hỏng → có fallback public 4 quý nhưng mất history).
- **Quy ước `release_date`**: GDP = quý +30 ngày, fundamentals = quý +45 ngày, CPI = month-end +6 ngày — đều bảo thủ. Step 2 dựa vào các mốc này để as-of join chống leakage.
- **Warmup YoY (A2)**: GDP & fundamentals fetch từ `WARMUP_START = 2017-01-01`. GDP có 2017-Q1 (37 quý). **Fundamentals: VCI không có dữ liệu TCB trước 2018-Q1** (TCB niêm yết 2018) → chỉ tới 2018-Q1 (33 quý); `*_growth_yoy`/`pe_ratio` NaN dẫn đầu tới ~2019 (P/E thiếu 2018-Q1 nên leading-NaN ~52 phiên; giới hạn đã biết — không làm xấu thêm vùng dùng được vì technical warmup ~252 phiên cũng tới ~giữa 2019). CPI (IMF) đủ `cpi_yoy` từ đầu.