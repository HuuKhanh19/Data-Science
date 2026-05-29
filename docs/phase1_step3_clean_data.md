# Step 3 (Phase 1) — Làm sạch dữ liệu & Dựng trục lịch HOSE

Tài liệu này mô tả những gì Bước 3 đã làm được và giải thích chi tiết mã nguồn. Bước 3 hiện thực khâu **"Làm sạch (Data Cleaning) & dựng trục lịch HOSE"** trong pipeline Step 2 của `data.md` — *bước đầu tiên thực sự chạm và biến đổi dữ liệu* (EDA Phase 0 chỉ đọc). Nó biến các nguồn daily rời rạc thành nền móng sạch, đặt trên một trục thời gian chung, sẵn sàng cho tích hợp ở Bước 4.

---

## 1. Mục tiêu, Input, Output

| | Nội dung |
| :--- | :--- |
| **Mục tiêu** | Làm sạch 3 nguồn **daily** và dựng **spine** = lịch HOSE thực; căn VNINDEX và USD/VND về spine. Đặt nền cho as-of join ở Bước 4. |
| **Input** | 3 file daily đã schema-locked từ Bước 1: `data/raw/{tcb_price, vnindex, usdvnd}.parquet` |
| **Output** | 3 file `data/interim/*_clean.parquet` + `data/interim/_clean_log.json` (audit + exit code) |
| **Lệnh chạy** | `python scripts/clean_phase1.py` |
| **Kết quả** (29/05/2026) | spine **1994** phiên (2018-06-04 → 2026-05-28); `vnindex_clean` 1994; `usdvnd_clean` 1994, **ffill=239**, **leading_nan=0**; 0 error |

**Vì sao chỉ 3/6 nguồn xuất interim.** Bước 3 chỉ đụng vào 3 nguồn **daily** vì chỉ chúng mới cần căn theo từng phiên HOSE. Ba nguồn chậm — `cpi` (tháng), `gdp` (quý), `tcb_fundamentals` (quý) — lệch tần suất, không có khái niệm "căn theo phiên". Việc đưa chúng lên trục daily chính là **as-of join theo `release_date`**, đó là nhiệm vụ của Bước 4. Chúng giữ nguyên ở `data/raw/` như đã validate, đúng quy ước đã chốt.

**Ranh giới (Bước 3 KHÔNG làm).** Không merge thành panel (Bước 4), không tính feature (Bước 5), không gán nhãn (Bước 6), không cắt warmup (Bước 7), không Z-score/chia walk-forward (Bước 8).

---

## 2. Kiến trúc & bản đồ file

```
scripts/clean_phase1.py          ← runner: đọc raw → clean + spine → ghi interim + log + exit code
└── src/data/
    ├── clean.py                 ← MỚI: logic thuần (clean_tcb_price, build_spine, align_vnindex, align_fx)
    ├── schema.py                ← TÁI DÙNG: TCB_PRICE_SCHEMA, VNINDEX_SCHEMA (.validate)
    └── validation.py            ← TÁI DÙNG: check_monotonic_dates
```

Tách hai lớp có chủ đích: **logic thuần** (`clean.py`, DataFrame vào → DataFrame ra, không đụng disk) và **I/O mỏng** (`clean_phase1.py`, đọc/ghi file). Nhờ vậy hàm unit-test được, gọi trực tiếp nối chuỗi được, và Phase 2 chỉ việc bọc orchestrator quanh các script — mỗi bước "file vào → file ra", resume được, audit được.

---

## 3. Quy tắc khóa thực thi ở Bước 3

Tất cả chốt từ EDA Phase 0 (xem `docs/phase1_step2_eda.md`):

| Hạng mục | Quyết định |
| :--- | :--- |
| Spine | Tập ngày giao dịch của `tcb_price` (lịch HOSE). **Không** tự sinh business-day (sẽ lệch nghỉ lễ VN). |
| TCB & VNINDEX | Ngày tăng nghiêm ngặt, `close > 0`, **không** ffill giá/return. |
| VNINDEX thiếu phiên spine | **Lỗi cứng (`raise`)** — chỉ số luôn được tính khi sàn mở, thiếu là lỗi data, không ffill. |
| FX (USD/VND) | Reindex về spine + **ffill mức tỷ giá** theo as-of backward; **không bao giờ bfill**; leading NaN để nguyên (cắt ở Bước 7). |
| Cờ `fx_ffilled` | Đánh dấu phiên HOSE không có quan sát FX gốc đúng ngày (để audit). |
| Macro (CPI/GDP/fund) | Giữ nguyên raw → as-of join ở Bước 4. |

**Vì sao ffill FX mà không ffill giá.** FX chạy lịch 24/5 khác HOSE; một phiên HOSE mở mà FX nghỉ thì *mức tỷ giá nhà đầu tư thực sự thấy* là mức gần nhất đã biết → ffill là mô phỏng production đúng (và %change ngày đó = 0, phản ánh "không có tin mới"). Ngược lại, mọi phiên trên spine đều là ngày sàn mở nên giá TCB/VNINDEX là quan sát thật — ffill giá sẽ đẻ ra return/nhãn giả, đúng kiểu leakage cần tránh.

---

## 4. Giải thích mã nguồn

### 4.1 `src/data/clean.py` — logic thuần

**Tiện ích nội bộ**
- `_normalize_dates(df)`: ép `date` về datetime naive (drop tz), về midnight; sort tăng dần + drop trùng ngày (`keep="last"`) + reset index.
- `_order_cols(df, extra)`: sắp cột theo `OHLCV_ORDER` (`date, open, high, low, close, volume, fetched_at`), nối thêm cột phụ nếu có (vd `fx_ffilled`).
- `_assert_positive_close(df, name)`: đếm `close <= 0`, có thì `raise` (bỏ qua NaN — leading NaN của FX xử lý riêng).

**`clean_tcb_price(tcb_raw)`** — làm sạch giá TCB: `_normalize_dates` → assert `close > 0` → trả OHLCV. Đây là nguồn chân lý của spine.

**`build_spine(tcb_clean)`** — trả `DatetimeIndex` (đặt tên `date`) từ ngày của TCB đã sạch, sort tăng dần; `raise` nếu có ngày trùng hoặc không tăng nghiêm ngặt. Mọi nguồn daily khác sẽ căn về index này.

**`align_vnindex(vnindex_raw, spine)`** — `_normalize_dates` → `set_index("date")` → `reindex(spine)` **không** `method` (ngày thiếu thành NaN). Nếu có phiên spine nào `close` NaN → `raise` kèm danh sách ngày phạm (lỗi data, không ffill). Ngược lại assert `close > 0` và trả OHLCV trên spine. Các ngày VNINDEX có nhưng *không* thuộc spine (vd phiên TCB tạm ngừng) tự rụng khi reindex.

**`align_fx(usdvnd_raw, spine)`** — điểm tinh tế nhất:
- `_normalize_dates` rồi `reindex(spine, method="ffill")`. `method="ffill"` lấy quan sát gần nhất **≤ mỗi phiên** *từ chuỗi FX gốc* (kể cả ngày không thuộc spine) → đúng nghĩa as-of backward, không phải chỉ ffill trong phạm vi spine.
- Leading NaN (phiên spine trước quan sát FX đầu tiên) **để nguyên** — `method="ffill"` không lấp ngược, đảm bảo không bfill.
- Cờ `fx_ffilled = (date không có trong FX gốc) & (close không NaN)` — tức phiên HOSE đã được lấp bằng giá trị quá khứ. Leading NaN không bị tính là ffill.

### 4.2 `scripts/clean_phase1.py` — runner

- `ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))` rồi import từ `src.data.*` — đồng bộ pattern với `scripts/fetch_phase1.py`.
- `main()`:
  1. Đọc 3 raw daily.
  2. **TCB + spine**: `clean_tcb_price` → `build_spine` → `TCB_PRICE_SCHEMA.validate` → `check_monotonic_dates` → ghi `tcb_price_clean.parquet`.
  3. **VNINDEX**: `align_vnindex` → `VNINDEX_SCHEMA.validate` → `check_monotonic_dates` → ghi `vnindex_clean.parquet`.
  4. **FX**: `align_fx` → đếm `n_ffilled` & `n_leading_nan` → ghi `usdvnd_clean.parquet` (cảnh báo nếu có leading NaN).
  5. Ghi `_clean_log.json`, in summary, trả **exit code 0/1** (tiện job tự động Phase 2).
- Toàn bộ bọc `try/except`: lỗi được ghi vào log thay vì làm sập im lặng.

---

## 5. Kiểm thử & nghiệm thu

Bốn check đã chạy (script `verify.py`, dữ liệu tổng hợp tái hiện đúng đặc tính rồi xác nhận lại trên dữ liệu thật):

1. **Đồng trục**: 3 file `_clean` có `date` trùng khít spine, tăng nghiêm ngặt, không trùng, `close > 0`.
2. **FX as-of**: mỗi phiên FX bằng đúng giá trị `merge_asof(direction="backward")` từ chuỗi gốc → ffill kéo từ ngày gốc gần nhất, không chỉ trong spine.
3. **Cờ & không bfill**: `fx_ffilled` khớp tuyệt đối; các NaN còn lại chỉ là **prefix** đầu chuỗi (không bfill, không lỗ giữa chuỗi).
4. **Path lỗi cứng**: bỏ 1 phiên spine khỏi VNINDEX → `align_vnindex` `raise` đúng quy ước.

Kết quả thật khớp EDA Phase 0 từng con số: spine **1994**, FX `ffill` **239** (= số ngày HOSE thiếu FX EDA đếm được), `leading_nan=0` (FX có quan sát ngay 2018-06-04).

---

## 6. Schema output

| File | Cột | Ghi chú |
| :--- | :--- | :--- |
| `tcb_price_clean.parquet` | `date, open, high, low, close, volume, fetched_at` | OHLCV đầy đủ; `date` = spine (1994). |
| `vnindex_clean.parquet` | `date, open, high, low, close, volume, fetched_at` | Căn về spine; cùng 1994 phiên. |
| `usdvnd_clean.parquet` | `... + fx_ffilled (bool)` | Căn spine + ffill mức; `fx_ffilled` đánh dấu 239 phiên. FX `volume` từ yfinance ~0, không dùng. |

Giữ **full OHLCV** (không chỉ `date`+`close`) để khỏi phải fetch lại nếu L2 cần `high/low/volume`. `data/interim/` nằm trong `.gitignore` (tái sinh được từ runner).

---

## 7. Tiếp theo — Bước 4 (Tích hợp)

Bước 4 đọc **3 file `data/interim/`** (daily, đã trên spine) + **3 nguồn `data/raw/`** chậm (CPI/GDP/fundamentals), rồi: daily merge VNINDEX & USD/VND theo `date`; **as-of join backward** L3/L4 theo `release_date ≤ t`; forward-fill giữa hai kỳ release. Đây là chốt chặn leakage số 2 trong bốn chốt của pipeline.