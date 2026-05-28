# Step 1 (Phase 1) — Thu thập Raw Data

Tài liệu này mô tả những gì Step 1 đã làm được và giải thích chi tiết mã nguồn. Step 1 hiện thực **khâu "thu thập"** trong pipeline data của `data.md`: lấy 6 nguồn raw, kiểm định, lưu thành parquet — *chưa* biến đổi (transform để dành Step 2).

---

## 1. Mục tiêu, Input, Output

| | Nội dung |
| :--- | :--- |
| **Mục tiêu** | Thu thập trung thực 6 nguồn dữ liệu (2018-06-04 → hiện tại), kiểm định chất lượng, lưu parquet schema-locked |
| **Input** | API/web ngoài: vnstock (VCI), yfinance, NSO CPI archive, VBMA GDP TSV |
| **Output** | 6 file `data/raw/*.parquet` + `data/raw/_fetch_log.json` (nhật ký + thống kê mỗi nguồn) |
| **Lệnh chạy** | `python scripts/fetch_phase1.py` |
| **Kết quả** (28/05/2026) | 6/6 OK — `tcb_price` 1994, `vnindex` 2086, `usdvnd` 2077, `cpi` 100, `gdp` 32, `tcb_fundamentals` 32 |

**Nguyên tắc thiết kế** (xuyên suốt mọi module):
- *Raw-only*: chỉ thu thập, không tính feature (YoY, log-return... để Step 2).
- *Schema lock*: mỗi output có hợp đồng cột/null/khóa chính, sai là raise ngay.
- *Anti-leakage từ gốc*: biến chậm lưu kèm `release_date` (ngày công bố thực).
- *Audit*: mọi bảng có `fetched_at` (giờ VN); CPI lưu cả raw text để truy vết.

---

## 2. Kiến trúc & bản đồ file

```
scripts/fetch_phase1.py          ← orchestrator: gọi 3 channel, ghi log + summary
└── src/data/
    ├── schema.py                ← "hợp đồng dữ liệu": 6 ParquetSchema + .validate()
    ├── validation.py            ← validators chất lượng cho dữ liệu giá
    ├── fetch_prices.py          ← Channel A: tcb_price, vnindex, usdvnd  (daily)
    ├── fetch_macro.py           ← Channel B: cpi (scrape), gdp (TSV)     (tháng/quý)
    └── fetch_fundamentals.py    ← Channel C: tcb_fundamentals            (quý)
```

Mỗi hàm `fetch_*` trả về một **dict báo cáo** đồng nhất: `{status, rows, date_min, date_max, output, ...}` để orchestrator tổng hợp.

---

## 3. Giải thích mã nguồn

### 3.1 `scripts/fetch_phase1.py` — Orchestrator

Điều phối toàn bộ. `START_DATE = "2018-06-04"` (ngày niêm yết TCB), `END_DATE` = hôm nay (giờ VN).

- **`_run_one(name, fn, **kwargs)`**: wrapper gọi một hàm fetch, in tiến trình, bắt mọi exception và quy về `{status:"error", ...}` để **một nguồn lỗi không làm sập cả pipeline**.
- **`main()`**: tạo `data/raw/`, lần lượt chạy Channel A → B → C, gom kết quả vào `results`, ghi `_fetch_log.json`, in summary (đếm OK/warning/error). Trả exit code `0` nếu không có error, `1` nếu có (tiện cho tự động hóa ở Phase 2).

### 3.2 `src/data/schema.py` — Hợp đồng dữ liệu

Định nghĩa cấu trúc bắt buộc của từng output, tách biệt khỏi logic fetch.

- **`ColumnSpec`**: tên cột, dtype, `nullable`, mô tả.
- **`ParquetSchema.validate(df)`** kiểm 3 điều, fail thì raise `ValueError`:
  1. **Đủ cột**: thiếu cột khai báo → lỗi.
  2. **Ràng buộc non-null**: cột `nullable=False` mà có null → lỗi (vd `close`, `reference_period`).
  3. **Khóa chính không trùng**: vd `date` (giá) hoặc `reference_period` (CPI/GDP/fundamentals) bị lặp → lỗi.
- 6 schema cụ thể: `TCB_PRICE / VNINDEX / USDVND` dùng chung khung **OHLCV** (`date, open, high, low, close, volume, fetched_at`, khóa `date`); `CPI` (có `raw_text`, `release_date`), `GDP` (`nominal_gdp_vnd_bil`, `release_date`), `TCB_FUNDAMENTALS` (10 cột chỉ số + `release_date`).

### 3.3 `src/data/validation.py` — Kiểm định chất lượng giá

Trả về `ValidationReport(warnings, errors)`. Bốn kiểm tra:

- **`check_monotonic_dates`**: ngày phải tăng nghiêm ngặt, không trùng (error nếu vi phạm).
- **`check_hose_calendar_gap`**: gap giữa 2 phiên > 12 ngày → **warning**, > 15 ngày → **error** (cho phép gap Tết ~7–10 ngày là hợp lệ).
- **`check_abnormal_returns`**: |log-return| ngày > 15% → cảnh báo (dấu hiệu adjustment chưa apply hoặc lỗi data); giá ≤ 0 → error. *Trên adjusted close đúng, TCB có 0 phiên vi phạm dù chia cổ tức cổ phiếu 1:1 năm 2024.*
- **`canonicalize_price_unit`**: median giá phải nằm trong khoảng kỳ vọng (TCB: [5, 200] nghìn VND); lệch → raise để **người kiểm tra**, *không tự convert* (an toàn hơn).

### 3.4 `src/data/fetch_prices.py` — Channel A (giá daily)

Hai hàm tiện ích dùng chung:
- **`_normalize_ohlcv(df, rename_map)`**: chuẩn hoá về khung OHLCV — reset index nếu ngày là index, **drop timezone** (yfinance tz-aware), bỏ cột thừa (`Adj Close, Dividends, Stock Splits`), lowercase tên cột, ép kiểu, **drop dòng `close` null** (yfinance đôi khi trả NaN cho phiên "hôm nay"), sort + drop trùng ngày, gắn `fetched_at`.
- **`_save_and_report(df, schema, validators)`**: chạy `schema.validate` → chạy các validator → ghi parquet → trả dict báo cáo.

Ba hàm thu thập:
- **`fetch_tcb_price`**: `vnstock.Quote(symbol="TCB", source="VCI").history(...)`. **VCI trả adjusted close** (đã xử lý corporate action). Chạy đủ 3 validator + `canonicalize_price_unit`.
- **`fetch_vnindex`**: tương tự nhưng **bỏ** `canonicalize_price_unit` (VN-Index ~800–1500, khác đơn vị) và bỏ `check_abnormal_returns` (chỉ số ít khi nhảy >15%).
- **`fetch_usdvnd`**: `yfinance.Ticker("USDVND=X")`. **Bỏ** `check_hose_calendar_gap` vì FX chạy lịch 24/5 khác HOSE.

### 3.5 `src/data/fetch_macro.py` — Channel B (CPI, GDP)

Có `_http_get` / `_http_get_bytes` (retry, User-Agent, tùy chọn `verify_ssl`).

**CPI — `fetch_cpi`** (scrape NSO `/en/cpi/`, lưu raw text):
- **Phân trang**: WordPress dùng `?paged=N` (đã verify từ HTML), lặp page 1..N.
- **`_parse_nso_page(html)`**: BeautifulSoup tìm `<section class="item">`; mỗi section lấy `title`, span `archive-issue-date` (→ `release_date` qua regex `Date of issue: DD/MM/YYYY`), span `archive-reference-period` (→ tháng tham chiếu). **Lưu `raw_text`** của body, *không* parse số — đúng tinh thần raw-only.
- **`_parse_reference_period`**: xử lý **2 định dạng** NSO dùng song song — text "April 2026" (bài mới) và số "8/2024" (bài cũ).
- **Lọc & dừng sớm**: tính `cutoff = start_date − 13 tháng` (đủ tiền sử để Step 2 tính YoY đầu kỳ). Vì NSO liệt kê giảm dần theo ngày, khi trang hiện tại đã cũ hơn cutoff thì **dừng** (tiết kiệm request). Dừng cả khi gặp trang rỗng hoặc toàn bản ghi trùng.
- Cuối cùng gom `DataFrame`, lọc theo `[cutoff, end_date]`, validate, lưu. `status="ok"` nếu ≥ 50 dòng, ngược lại `"warning"`.

**GDP — `fetch_gdp`** (VBMA TSV):
- Tải bytes, **dò encoding**: file là **UTF-16 LE** (BOM `\xff\xfe`) — code thử lần lượt `utf-16, utf-8-sig, utf-8, cp1252, latin-1` và chỉ chấp nhận encoding nào parse ra **cột dạng "Q\<n\> YYYY"** (chặn latin-1 decode "thành công" nhưng ra ký tự rác).
- File là **TSV** (`sep="\t"`), layout B: cột = nhãn quý, một hàng nhãn "Nominal Gross Domestic Product". Code tìm đúng hàng đó, parse giá trị dạng `"809,613"` (bỏ ngoặc kép + dấu phẩy ngăn nghìn) → list `(quarter_end, value)`.
- Gán `release_date = quarter_end + 30 ngày` (quy ước bảo thủ vì VBMA không cung cấp ngày công bố), lọc theo khoảng, validate, lưu.

### 3.6 `src/data/fetch_fundamentals.py` — Channel C (cơ bản TCB)

Điểm mấu chốt: API public của vnstock **chỉ trả 4 quý**. Code **gọi thẳng method private** để mở khoá full history.

- **`_fetch_full_history(symbol)`**: import `vnstock.explorer.vci.financial.Finance` (VCIFinance), gọi `_get_financial_report(stmt_type, period="quarter", lang="en", get_all=True, limit=100)` cho `balance_sheet / ratio / income_statement` → mỗi statement ~31–33 quý. Có **fallback** về method public (4 quý) nếu private call lỗi. (`redirect_stdout` để chặn banner quảng cáo vnstock.)
- **`FIELD_MAPPING`**: ánh xạ tên feature → `(statement, [item_id ưu tiên])`, vd `credit_balance → loans_and_advances_to_customers_net`, `npl_ratio_pct → npl`, `nim_pct → net_interest_margin`.
- **`_find_item_row` / `_extract_quarter_series`**: tìm đúng dòng theo `item_id`, trích các cột dạng `YYYY-Qn` thành chuỗi `{quarter_end: value}`.
- **`_normalize_to_billion`**: nếu giá trị quá lớn (median > 1e9, tức đơn vị VND) thì chia 1e9 về **tỷ VND**; bỏ qua với cột `_pct` và `eps`.
- **`fetch_tcb_fundamentals`**: ghép tất cả field theo `reference_period`, cột `interest_earning_assets` để `NA` (vnstock không có), gán `release_date = quý + 45 ngày`, sắp đúng thứ tự cột theo schema, validate, lưu. Báo cáo kèm `null_counts` và `field_sources` (truy vết mỗi feature lấy từ item_id nào).

---

## 4. Lưu ý & giới hạn đã biết

- **`.gitignore`**: `data/raw/*.parquet`, `*.csv`, `_fetch_log.json` không commit (regenerate được bằng script). Thư mục `_debug/` và các script chẩn đoán đã được dọn ở bước clean.
- **Phụ thuộc nguồn ngoài**: CPI/GDP là scrape — nếu NSO/VBMA đổi layout, hàm sẽ raise lỗi rõ ràng (cần cập nhật parser). Đây là rủi ro cần theo dõi khi auto-refit ở Phase 2.
- **Quy ước `release_date`**: GDP +30 ngày, fundamentals +45 ngày (bảo thủ); CPI dùng *Date of issue* thực. Step 2 dựa vào các mốc này để as-of join.
- **CPI lưu raw text**: việc trích MoM → ghép YoY là của Step 2, không phải Step 1.