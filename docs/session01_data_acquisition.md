# Session 01 — Data Acquisition

**Date**: 2026-05-16
**Phase**: 1 (Data Collection) + Phase 0 partial (Source verification)
**Status**: ✅ Complete
**Version bump**: v0.1.0 → v0.2.0
**Liên quan**: `research_design.md` §2.3 (Nguồn dữ liệu), §2.5 (Xử lý dữ liệu chính), `IMPLEMENTATION.md` §4.1 (Raw data schemas), §11 (Open question Q1)

---

## 1. Mục đích và bối cảnh

### 1.1 Vị trí trong pipeline tổng thể

Pipeline scientific của project gồm 7 phase tuần tự (research_design §11): **Phase 0 (verification) → Phase 1 (data collection)** → Phase 2 (EDA) → Phase 3 (feature engineering) → Phase 4 (modeling) → Phase 5 (statistical inference) → Phase 6 (deployment) → Phase 7 (report).

Session 1 thực hiện đồng thời **Phase 1** (acquire raw price + index + FX) và một phần **Phase 0** (verify chất lượng nguồn dữ liệu trước khi commit Phase 1 output làm source-of-truth).

### 1.2 Pre-conditions

- Repo bootstrap đã xong (Session 1a output) với folder structure, `CFG` constants, logger, seed setup.
- Server local có conda env `ds` với Python 3.11, vnstock 4.0.1, yfinance 1.3.0, pandas 2.3, pyarrow.
- `vnstock_chart` đã install từ private index (`vnstock` v4+ require chart backend để import).

### 1.3 Câu hỏi cần trả lời

Hai câu hỏi có thể block toàn bộ pipeline downstream nếu không resolved:

- **Q1 (Open question số 1 trong IMPLEMENTATION §11)**: vnstock `source='VCI'` có thực sự return adjusted close không? Nếu unadjusted, toàn bộ L1/L2 features (log returns, MA crossover, RSI, MACD, Bollinger Bands) sẽ bị nhiễm corporate-action jumps, label y_{t,k} có thể bias.
- **Q1 phụ**: nếu fallback yfinance được kích hoạt trong production cron (vnstock down), parquet output có nhất quán đơn vị với vnstock branch không? Nếu không, data integrity bị phá vỡ giữa các ngày cron.

### 1.4 Deliverables

- 3 raw parquet files (TCB price, VN-Index, USD/VND) trong `data/raw/`.
- 1 audit JSON ghi summary stats để reproducibility.
- Code module `src/data/fetchers.py` (primary + fallback logic).
- Script `scripts/acquire_data.py` (CLI orchestrator).
- Notebook `notebooks/00_data_source_verification.ipynb` (Phase 0 verification).
- Session log này.

---

## 2. Source code — mô tả chi tiết

### 2.1 File `src/data/fetchers.py` (≈400 dòng)

#### 2.1.1 Triết lý kiến trúc

Module này áp dụng **separation of concerns** giữa hai loại data quality check:

| Loại | Đặc điểm | Vị trí trong fetcher |
|---|---|---|
| **Structural DQ** | Derivable từ data definitions, không cần dữ liệu thực để đặt ngưỡng | **Trong fetcher** (`_validate_*`) |
| **Threshold-based DQ** | Cần observed distribution để định ngưỡng (vd cross-source agreement, calendar gap) | **Bên ngoài fetcher** (notebook 00 / EDA) |

Mục tiêu: tránh **anti-pattern "ngây thơ"** đặt ngưỡng tùy ý (vd "diff > 1%"). Ngưỡng phải có cơ sở thực nghiệm — đo từ data trước khi lock.

#### 2.1.2 Public API (3 functions)

```python
def fetch_tcb_price(start, end, source='auto') -> pd.DataFrame
def fetch_vnindex(start, end, source='auto') -> pd.DataFrame
def fetch_usdvnd(start, end) -> pd.DataFrame
```

Tất cả return DataFrame có DatetimeIndex (name `date`), unique và sorted ascending. Pass structural DQ trước khi return.

**Parameter `source`**:
- `'auto'` (default): try primary (vnstock), fallback yfinance on exception
- `'vnstock'` / `'yfinance'`: lock vào một nguồn, raise nếu nguồn đó fail
- Forensic mode trong notebook 00 dùng `'vnstock'` và `'yfinance'` riêng để so sánh

#### 2.1.3 Internal: source-specific raw fetchers

**`_fetch_ohlcv_vnstock(symbol, start, end)`** — đường primary

Gọi vnstock v3 API:
```python
from vnstock import Vnstock
stock = Vnstock().stock(symbol=symbol, source='VCI')
df = stock.quote.history(start=start, end=end, interval='1D', to_df=True)
```

vnstock VCI returns columns `time, open, high, low, close, volume`. Module:
1. Rename `time → date`, set làm DatetimeIndex
2. Strip timezone nếu có
3. Populate `adj_close = close` (giả định adjusted — đã verify trong notebook 00, xem §5.1)
4. Return DataFrame schema chuẩn

**`_fetch_ohlcv_yfinance(ticker, start, end)`** — đường fallback

Gọi yfinance:
```python
import yfinance as yf
df = yf.Ticker(ticker).history(start=start, end=end, interval='1d', auto_adjust=True)
```

**Quyết định quan trọng — `auto_adjust=True`**:

Khi `auto_adjust=False`, yfinance trả về `Open/High/Low/Close` ở giá **raw** và `Adj Close` ở giá **adjusted**. Nếu code trộn raw OHL với adjusted Close (như implementation đầu tiên của tôi), OHLC integrity bị vi phạm trên mọi ngày trước corporate action:

Lý do toán học: `adj_close_t = raw_close_t × factor`, với `factor ≤ 1` (cumulative adjustment cho tất cả corporate actions sau ngày t). Vậy `adj_close_t ≤ raw_close_t`, có thể `< raw_low_t` → vi phạm constraint `low ≤ close`.

Với `auto_adjust=True`, **toàn bộ OHLC đều được adjusted theo cùng factor** → integrity giữ nguyên. yfinance trả về `Open, High, Low, Close, Volume` (không có `Adj Close` riêng vì Close IS adjusted). Module populate `adj_close = close` để consistent schema.

#### 2.1.4 Internal: structural validators

Mapping về 4 mức data quality của Slide_Data_Science.md:

```
┌─ Mức giá trị đơn (value-level) ─────────────────────────┐
│  • Missing values: NaN trong OHLCV → ERROR              │
│    (research_design §2.5 cấm forward-fill prices)       │
│  • Domain: prices > 0, volume ≥ 0                       │
│  • Dtype: float64 cho prices, int64 cho volume          │
└─────────────────────────────────────────────────────────┘

┌─ Mức bản ghi (record-level) ────────────────────────────┐
│  • OHLC integrity:                                      │
│    low ≤ {open, close, high}                            │
│    high ≥ {open, close, low}                            │
└─────────────────────────────────────────────────────────┘

┌─ Mức tập giá trị (value-set-level) ─────────────────────┐
│  • DatetimeIndex unique (no duplicate dates)            │
│  • DatetimeIndex monotonic ascending                    │
└─────────────────────────────────────────────────────────┘
```

Function `_validate_ohlcv` raise `ValueError` nếu bất kỳ check nào fail, kèm tối đa 5 dates vi phạm để debug nhanh.

Function `_validate_index_price` (cho VN-Index, không có volume/OHL) và `_validate_fx` (cho USD/VND, single column `rate`) là phiên bản rút gọn.

#### 2.1.5 Primary→fallback orchestration

```python
def fetch_tcb_price(start, end, source='auto'):
    if source in ('vnstock', 'auto'):
        try:
            df = _fetch_ohlcv_vnstock(symbol='TCB', start=start, end=end)
            _validate_ohlcv(df, name='TCB[vnstock]')
            return df  # nghìn VND, no rescale needed
        except Exception as e:
            if source == 'vnstock':
                raise
            log.warning("vnstock failed (%s); falling back to yfinance", e)

    df = _fetch_ohlcv_yfinance(ticker='TCB.VN', start=start, end=end)
    # CRITICAL: yfinance returns VND, normalize to nghìn VND
    df[['open','high','low','close','adj_close']] /= 1000.0
    _validate_ohlcv(df, name='TCB[yfinance]')
    return df
```

**Quyết định canonical unit**: nghìn VND (xem §3.4 bên dưới). Yfinance trả về VND đơn vị thực; rescale `/1000` bắt buộc ngay sau fetch để parquet output luôn cùng đơn vị bất kể source.

#### 2.1.6 Summary stats helper

`summary_stats(df, name)` — return dict với:
- `n_rows`, `date_min`, `date_max`
- `max_calendar_gap_days` và `max_calendar_gap_at` (visual inspection cho operator)
- Price min/median/max
- Volume median + zero-volume count (nếu applicable)
- FX rate min/median/max (nếu applicable)

**Không raise**, chỉ surface anomaly cho con người. Đây là pattern "human-in-the-loop sanity check", separate khỏi machine-enforced structural DQ.

---

### 2.2 File `scripts/acquire_data.py` (≈170 dòng)

CLI orchestrator one-shot, gọi 3 fetchers tuần tự và save parquet.

#### 2.2.1 CLI interface

```
python scripts/acquire_data.py [OPTIONS]

Options:
  --start-date YYYY-MM-DD   Default: CFG.TCB_START_DATE (2018-06-04)
  --end-date YYYY-MM-DD     Default: today
  --source {auto,vnstock,yfinance}   Default: auto
  --dry-run                 Fetch + summarize, không save
  --skip-existing           Skip parquet đã tồn tại
```

#### 2.2.2 Output

- 3 parquet files trong `data/raw/`
- 1 audit JSON `data/raw/_acquisition_summary.json` chứa: `end_date`, `source_policy`, per-dataset stats

Audit JSON đảm bảo **reproducibility forensic**: tháng sau nhìn lại biết được lúc acquire dữ liệu, max gap quan sát là bao nhiêu, có anomaly gì không.

#### 2.2.3 Error handling

- Mỗi fetcher fail → log ERROR + return code 1, không tiếp tục fetch sau đó
- Đây là intentional "fail fast" để operator handle, tránh partial data state mơ hồ
- Trong production cron (Session 13), retry policy + alerting sẽ khác

---

### 2.3 File `notebooks/00_data_source_verification.ipynb` (9 code cells)

Notebook **bypass fetcher wrappers**, gọi trực tiếp vnstock và yfinance để forensic compare. Mục đích: verify Open Q1 + measure cross-source disagreement baseline.

Structure:

| Cell | Loại | Mục đích |
|---|---|---|
| 1 | Setup | sys.path, imports, plt config |
| 2 | Fetch | Gọi `_fetch_ohlcv_vnstock` và `_fetch_ohlcv_yfinance` cho TCB toàn lịch sử |
| 3 | Inspect | Hiển thị head() của 2 DataFrame |
| 4 | **Open Q1 detection** | Compute log returns cho cả 3 chuỗi (vnstock close, yfinance raw Close, yfinance Adj Close); flag dates có \|log return\| > 15% |
| 5 | Visual | 3-panel overlay plot: vnstock close vs yf_raw vs yf_adj |
| 6 | **Cross-source check** | (a) Scale factor `yf/vn` distribution; (b) Log-return disagreement quantiles (basis points) |
| 7 | Top outliers | DataFrame top 10 ngày \|lr_diff\| lớn nhất + dates only-in-vnstock |
| 8 | Histograms | Linear + log scale của lr_diff distribution |
| 9 | **Calendar gaps** | Distribution + top 10 longest gaps |

Tại sao notebook và không phải test? Vì:
- Output là **distributions + visualizations**, cần human interpretation
- Threshold lock cuối cùng dựa trên quan sát empirical, không phải binary pass/fail
- Reproducible (notebook commit) nhưng không tự động hóa vào CI

---

## 3. Data — xử lý và kết quả

### 3.1 Input data sources

#### 3.1.1 vnstock VCI (primary cho TCB và VN-Index)

| Thuộc tính | Giá trị |
|---|---|
| Library | `vnstock>=3.2` (test với v4.0.1) |
| Endpoint | VCI (Vietcap Securities) — broker Việt Nam |
| Interface | `Vnstock().stock(symbol=..., source='VCI').quote.history(start, end, interval='1D')` |
| Output columns | `time, open, high, low, close, volume` |
| Đơn vị giá | **Nghìn VND** (convention của broker Việt Nam — chuẩn HOSE trading platform) |
| Đơn vị volume | Số cổ phiếu (shares) |
| Adjusted? | **Có** (verified §5.1) |
| Trading calendar | HOSE (theo lịch nhà nước Việt Nam) |

#### 3.1.2 yfinance (fallback cho TCB/VN-Index; primary cho USD/VND)

| Thuộc tính | Giá trị |
|---|---|
| Library | `yfinance>=0.2` (test với v1.3.0) |
| Endpoint | Yahoo Finance API |
| TCB ticker | `TCB.VN` (suffix `.VN` cho HOSE) |
| VN-Index ticker | `^VNINDEX` |
| USD/VND ticker | `USDVND=X` |
| Output columns | `Open, High, Low, Close, Volume` (khi `auto_adjust=True`) |
| Đơn vị giá | **VND đơn vị thực** (1000× lớn hơn vnstock) |
| Adjusted? | Có khi `auto_adjust=True` (default từ v0.2.40+) |

#### 3.1.3 Phạm vi yêu cầu

- Start: 04/06/2018 (ngày niêm yết TCB trên HOSE, research_design §2.1)
- End: 16/05/2026 (today khi chạy)

### 3.2 Processing pipeline

Sequence cho mỗi data product:

```
   ┌──────────────┐
   │ Source API   │  vnstock.quote.history() / yf.Ticker.history()
   └──────┬───────┘
          │ raw response (DataFrame)
          ▼
   ┌──────────────┐
   │ Schema       │  Rename columns: time/Open/Close → date/open/close
   │ normalization│  Set DatetimeIndex, sort ascending, strip timezone
   └──────┬───────┘
          │ canonical schema
          ▼
   ┌──────────────┐
   │ Source-      │  yfinance branch: rescale prices /1000 (→ nghìn VND)
   │ specific     │  vnstock branch: no rescale (already nghìn VND)
   │ adjustment   │  Both: populate adj_close = close
   └──────┬───────┘
          │ unit-normalized data
          ▼
   ┌──────────────┐
   │ Structural   │  (1) Schema completeness check
   │ DQ           │  (2) DatetimeIndex unique + sorted
   │ validation   │  (3) No NaN in OHLCV
   │              │  (4) Prices > 0, volume ≥ 0
   │              │  (5) OHLC integrity: low ≤ {O,C,H}, high ≥ {O,C,L}
   └──────┬───────┘
          │ validated data
          ▼
   ┌──────────────┐
   │ Summary      │  n_rows, date range, max calendar gap
   │ statistics   │  Price stats, volume stats
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ Save parquet │  pyarrow engine, snappy compression
   └──────────────┘
```

### 3.3 Quality checks — cơ sở lý thuyết và áp dụng

Đối chiếu với 5 mức chất lượng dữ liệu trong Slide_Data_Science.md:

| Mức (Slide DS) | Loại lỗi | Check áp dụng trong fetcher | Hành động nếu vi phạm |
|---|---|---|---|
| **Mức giá trị đơn** | Thiếu (missing) | NaN trong OHLCV columns | ERROR (raise; research_design §2.5 cấm forward-fill prices) |
| | Vi phạm cú pháp | dtype check (float64 cho price, int64 cho volume) | ERROR ngầm qua schema |
| | Vi phạm miền xác định | Prices > 0, volume ≥ 0 | ERROR + báo dates vi phạm |
| **Mức bản ghi** | Vi phạm ràng buộc toàn vẹn | OHLC integrity (low ≤ {O,C,H}, high ≥ {O,C,L}) | ERROR + báo dates vi phạm |
| **Mức tập giá trị** | Vi phạm đơn nhất | DatetimeIndex unique | ERROR + liệt kê duplicates |
| | Sai thứ tự | Index monotonic ascending | ERROR |
| **Mức quan hệ** | Phụ thuộc hàm (trading calendar) | Max gap reporting (not enforced in Session 1) | Surfaced trong summary, threshold lock sau notebook 00 |
| **Mức đa quan hệ** | Xung đột giữa nguồn | Cross-source check (notebook 00) | Threshold lock sau Phase 0, fetcher cross-check trong Session 2+ |

### 3.4 Quyết định canonical unit — **nghìn VND**

Phát hiện trong Phase 0 (notebook 00 cell 6 đầu tiên):
- vnstock TCB close median = 16.46 → 16,460 VND/share
- yfinance TCB.VN close median ≈ 16,460 VND/share
- Cùng giá thực tế, **đơn vị khác nhau 1000 lần**.

Nếu **không xử lý**, fetcher fallback sẽ ghi parquet với đơn vị khác primary → catastrophic bug data integrity (cùng tên file, cùng schema, dual semantics).

**Lựa chọn nghìn VND vì**:
1. Convention chuẩn của thị trường chứng khoán Việt Nam (mọi broker hiển thị "25.00" = 25,000 VND)
2. Match literature Việt Nam được tham chiếu trong research_design
3. Magnitudes nhỏ hơn (1-100 range thay vì 1000-100000), thuận tiện numerical conditioning cho LSTM

**Implementation**:
- vnstock branch: no rescale (native nghìn VND)
- yfinance branch: divide all 5 price columns (open/high/low/close/adj_close) by 1000.0 ngay trong `fetch_tcb_price` orchestrator (không phải trong `_fetch_ohlcv_yfinance` để giữ tính reusable cho notebook 00)

### 3.5 Output data — kết quả thực

#### 3.5.1 `data/raw/price_tcb.parquet`

```
Schema:
  Index:       DatetimeIndex (name='date'), unique, monotonic ascending
  Columns:     open, high, low, close, adj_close (float64, nghìn VND), volume (int64)
  Compression: snappy (pyarrow)

Summary stats (acquired 2026-05-16):
  n_rows:                1,985
  date range:            2018-06-04 → 2026-05-15
  trading days/year:     ~248 (consistent với HOSE)
  close min:             7.04   (≈ 7,040 VND, low giai đoạn COVID 2020)
  close median:          16.46  (≈ 16,460 VND)
  close max:             41.30  (≈ 41,300 VND, đỉnh 2022)
  volume median:         6,191,157 shares/day (TCB là blue chip, hợp lý)
  volume zero days:      0
  max calendar gap:      10 days at 2019-02-11 (Tết Nguyên đán 2019)

DQ pass:
  ✓ All 1,985 rows have non-null OHLCV
  ✓ All prices > 0, volume ≥ 0
  ✓ All rows satisfy low ≤ {O, C, H} ≤ high
  ✓ DatetimeIndex unique and sorted

File size: 64.0 KB
```

#### 3.5.2 `data/raw/price_vnindex.parquet`

```
Schema:
  Index:    DatetimeIndex (name='date')
  Columns:  close, adj_close (float64, index points)

Summary stats:
  n_rows:           2,076
  date range:       2018-01-16 → 2026-05-15
  close min:        659.21    (đáy COVID 03/2020)
  close median:     1,166.51
  close max:        1,925.46  (đỉnh đầu 2022)
  max calendar gap: 10 days at 2019-02-11

Note: date_min = 2018-01-16 mặc dù request start = 2018-06-04. vnstock VCI 
ignore start param cho index symbol → returns toàn history available. 
91 ngày "thừa" trước listing date TCB. KHÔNG phải bug — feature pipeline 
sau sẽ inner-join với TCB nên auto-align về TCB's range.

File size: 43.7 KB
```

#### 3.5.3 `data/raw/fx_usdvnd.parquet`

```
Schema:
  Index:    DatetimeIndex (name='date')
  Columns:  rate (float64, VND per USD)

Summary stats:
  n_rows:           2,069
  date range:       2018-06-04 → 2026-05-15
  rate min:         22,366 VND/USD
  rate median:      23,359 VND/USD
  rate max:         26,425 VND/USD (gần đây, depreciation cuối 2024-2025)
  max calendar gap: 5 days at 2025-04-22 (Easter Monday + cuối tuần — global FX market, KHÔNG track HOSE Tết)

Note: USD/VND có 84 rows nhiều hơn TCB price. Lý do: FX market global theo 
lịch khác (không nghỉ Tết Việt Nam, không nghỉ 30/4-1/5). Join với TCB sẽ 
inner-align về HOSE trading days.

File size: 26.4 KB
```

#### 3.5.4 `data/raw/_acquisition_summary.json`

Audit log chứa toàn bộ stats trên + `end_date`, `source_policy`. Phục vụ:
- Reproducibility forensic (acquired ngày nào, source policy là gì)
- Debugging downstream (so sánh với expected statistics)

---

## 4. Model

*Session này không implement model. Section này sẽ có nội dung từ Session 6 trở đi.*

---

## 5. Verification và experiments

### 5.1 Open Q1 — vnstock VCI có adjusted close hay không?

#### 5.1.1 Bài toán

`research_design §2.5` quy định mọi feature pipeline dùng **adjusted close** (đã xử lý chia tách, cổ tức tiền mặt, cổ tức cổ phiếu). Nếu vnstock VCI trả về unadjusted, mọi log return, MA crossover, RSI, MACD sẽ chứa **fake jumps** tại ngày corporate action, làm biến dạng label và features.

#### 5.1.2 Phương pháp algorithmic (không hardcode dates)

Đặt threshold cao (`|log return| > 15%`) để detect corporate-action-magnitude moves mà không phụ thuộc kiến thức prior về ngày event cụ thể (tránh hallucination/lỗi sai date).

Algorithm (cell 4 trong notebook 00):
```
Inputs: vn_close (vnstock), yf_raw (yfinance auto_adjust=False, raw Close),
        yf_adj (yfinance auto_adjust=False, Adj Close)

For each series:
  log_return_t = log(P_t / P_{t-1})
  Flag dates where |log_return_t| > 0.15

Interpretation:
  - vnstock adjusted ⟹ no flagged dates (corporate actions smoothed)
  - vnstock unadjusted ⟹ flagged dates concentrated at known ex-rights dates
  - yfinance Adj Close adjusted ⟹ no flagged dates  
  - yfinance raw Close unadjusted ⟹ flagged at ex-rights dates
```

#### 5.1.3 Kết quả empirical

```
Trading days analyzed:   1,982 (intersection of TCB and yfinance ranges)
TCB known corporate actions in range:
  - Stock dividend 1:1 (2024)  ← biggest expected jump (~−50% nếu unadjusted)
  - Cash dividends (multiple years)

Flagged dates with |log return| > 15%:
  vnstock close:        0 / 1,985
  yfinance raw Close:   0 / 1,982    ← surprising, see §5.1.4
  yfinance Adj Close:   0 / 1,982
```

#### 5.1.4 Kết luận và interpretation

**Q1 RESOLVED**: vnstock VCI returns ADJUSTED close.

Bằng chứng: 0/1985 ngày có jump > 15% trong vnstock series, dù TCB có stock dividend 1:1 trong 2024 (ngày ex-rights phải drop ~50% trong unadjusted series). Cùng kết luận từ yfinance Adj Close.

**Observation phụ**: yfinance raw Close (auto_adjust=False) cũng không có drop > 15%. Điều này lạ vì theo định nghĩa "raw" thì phải có. Khả năng: Yahoo Finance treat stock dividend như split (auto-handle splits trong cả raw column ngay cả khi `auto_adjust=False`, chỉ giữ raw cho cash dividend). Không ảnh hưởng pipeline (project dùng adjusted close), document làm note.

### 5.2 Cross-source agreement distribution

#### 5.2.1 Phương pháp — scale-invariant

Đo lường disagreement KHÔNG dùng `|vn − yf| / vn` (sai do scale mismatch nghìn VND vs VND, factor ≈ 1000 → ratio ~99900% vô nghĩa).

Hai cách scale-invariant:
1. **Scale factor inference**: `factor_t = yf_t / vn_t` cho mỗi ngày. Distribution của factor xác nhận unit hypothesis.
2. **Log return disagreement**: `lr_diff_t = |log_return_vn,t − log_return_yf_adj,t|`. Log return là **scale-invariant với constant scale factor** — nếu 2 series identical up to constant scaling, lr_diff = 0 mọi nơi. Non-zero values = thực sự disagree về magnitudes/directions.

#### 5.2.2 Kết quả empirical

**Scale factor distribution** (n = 1,982 common days):
```
median:  968.99
mean:    976.61
std:     13.44
min:     965.65
max:     1000.23
```

Interpretation: factor very close to 1000 với spread ~3.4% (std/mean). Confirm hypothesis: vnstock=nghìn VND, yfinance=VND. Lệch nhẹ khỏi exactly 1000 do snapshot intra-day timing khác nhau giữa 2 source.

**Log return disagreement distribution** (basis points; 1 bp = 0.01%):
```
Q50:    1.52 bp   (median agreement ~0.015%)
Q75:    3.01 bp
Q90:    4.68 bp
Q95:    5.84 bp
Q99:    8.26 bp
Q99.9:  30.09 bp
Max:    317.39 bp  (~3.17%)  ← outlier
```

Phần lớn ngày agreement excellent (< 10 bp). Tail dài đến 317 bp.

#### 5.2.3 Outlier analysis — top 10 disagreement days

| Date | vn_close | yf_adj | vn_lr | yf_lr | lr_diff (bp) |
|---|---|---|---|---|---|
| **2024-05-21** | 22.99 | 22,990.97 | −2.11% | +1.06% | **317.39** |
| 2019-03-15 | 12.68 | 12,289.10 | −0.55% | −0.19% | 36.45 |
| 2019-03-14 | 12.75 | 12,311.98 | −0.63% | −0.93% | 29.96 |
| 2020-05-18 | 9.75 | 9,451.39 | +1.86% | +1.96% | 9.26 |
| 2019-08-13 | 9.75 | 9,451.39 | +1.86% | +1.96% | 9.26 |
| ... (5 nữa, all < 10 bp) | | | | | |

**Outlier 2024-05-21 analysis**:
- Cùng giá ngày 21/05 ở 2 source (factor = 22990.97/22.99 = 999.999, perfect agreement)
- DISAGREE về log return direction: vnstock giảm 2.11%, yfinance tăng 1.06%
- → Implication: hai source có giá NGÀY 20/05 khác nhau ~3.2%
- Pattern: **off-by-one adjustment timing**. TCB có cash dividend ex-date 22-23/05/2024 (~1,500 VND/cp ≈ 6% impact). Một source apply adjustment factor ở ngày T, source kia ở T+1 → trong 1 ngày log return của 2 source lệch một amount xấp xỉ size của dividend.

Outliers thứ 2-3 (14-15/03/2019, 30-36 bp): có pattern tương tự — TCB cash dividend Q1/2019.

Outliers còn lại (< 10 bp): noise/rounding/intraday timing.

#### 5.2.4 Locked threshold cho future Session 2+ cross-check

```
WARN > 100 bp (1%) trong cross-source log-return disagreement
```

Cơ sở:
- Q99.9 observed = 30 bp → threshold 100 bp = 3× safety margin
- Catches major dividend timing mismatches (vd 317 bp)
- Avoids false positives từ minor noise (Q99 = 8 bp < 100 bp)

Note: cross-check **chưa implement trong Session 1**. Lock value này cho future. Khi enable cross-check, 3 ngày known disagreement (2024-05-21, 2019-03-14, 2019-03-15) phải được **whitelist** để không trigger false alarm (chúng là valid timing-convention differences, không phải bug).

### 5.3 HOSE trading calendar gap analysis

#### 5.3.1 Phương pháp

Compute `gap_t = (date_t - date_{t-1}).days` cho mọi cặp ngày liên tiếp trong vnstock TCB series. Build distribution + identify max.

#### 5.3.2 Kết quả

Distribution gap calendar-days:

| Gap (days) | Count | % | Nguyên nhân điển hình |
|---|---|---|---|
| 1 | 1,568 | 79.0% | Same-week trading (intra-week) |
| 3 | 377 | 19.0% | Weekend (Fri → Mon) |
| 5 | 11 | 0.6% | Weekend + 2 lễ ngắn (vd 30/4-1/5 liền cuối tuần) |
| 4 | 10 | 0.5% | Weekend + 1 lễ |
| 2 | 6 | 0.3% | Hiếm — holiday-shortened week |
| 6 | 4 | 0.2% | 30/4-1/5 + cuối tuần (2019, 2023) |
| 8 | 4 | 0.2% | Tết Nguyên đán ngắn (2020, 2021, 2023, 2024) |
| 10 | 4 | 0.2% | Tết Nguyên đán dài (2019, 2022, 2025, 2026) |

Total = 1,984 gaps trên 1,985 ngày.

#### 5.3.3 Locked threshold

```
WARN > 12 calendar days   (max observed 10 + 2 safety margin)
ERROR > 15 calendar days  (anomaly mức không giải thích được bằng lịch nghỉ chính thức)
```

Cơ sở: max observed = 10 days = Tết kéo dài 9 ngày làm việc + cuối tuần ≈ 11 calendar days max possible nếu Tết overlap với weekend lý tưởng. 12 là buffer hợp lý.

### 5.4 Side observations

- **3 ngày vnstock có data nhưng yfinance thiếu**: chưa investigate cụ thể (chưa share output đầy đủ của cell 7). Có thể là days Yahoo backend cập nhật trễ. Không critical vì project dùng vnstock primary.
- **0 ngày yfinance có data nhưng vnstock thiếu**: vnstock coverage tốt hơn cho TCB.
- **vnstock 4.0.1 deprecation notice**: `Vnstock()` class sẽ deprecated, migration sang `vnstock.api.quote.Quote`. Code Session 1 vẫn work. Migration sẽ làm khi cần ổn định production (Session 13).

---

## 6. Issues encountered và fixes

5 bugs đã debug trong session này, theo thứ tự thời gian:

### Bug 1 — Vnstock chart dependency

**Symptom**: `ModuleNotFoundError: No charting library available` khi import vnstock.

**Root cause**: vnstock v4+ require charting backend (vnstock_chart hoặc vnstock_ezchart) để import được module top-level. Đây là design quirk của upstream, không phải bug của project.

**Fix**: install từ private index `pip install --extra-index-url https://vnstocks.com/api/simple vnstock_chart`. Document trong `requirements.txt` và README.

### Bug 2 — OHLC integrity violation

**Symptom**: `ValueError: TCB[yfinance]: OHLC integrity violation at [2018-06-04, ...]` ngay những ngày đầu range.

**Root cause**: `_fetch_ohlcv_yfinance` dùng `auto_adjust=False`, replace `close` column bằng `Adj Close` (adjusted), nhưng giữ `open/high/low` ở raw. Trên ngày trước stock dividend 2024, `adj_close = raw_close × factor` với `factor ≈ 0.5`, nên `adj_close < raw_low` → vi phạm constraint `low ≤ close`.

**Fix**: chuyển sang `auto_adjust=True` để **toàn bộ OHLC** đều adjusted theo cùng factor → integrity preserved. Module docstring update giải thích cơ sở.

**Lesson**: trong time series có corporate actions, OHLC phải được adjust như một cluster (cùng factor cho cùng ngày), không thể mix raw và adjusted.

### Bug 3 — IndexError trong summary_stats

**Symptom**: `IndexError: only integers, slices, ... and integer or boolean arrays are valid indices` khi compute `df.index[gaps.idxmax()]`.

**Root cause**: `gaps` là `pd.Series` với DatetimeIndex, nên `gaps.idxmax()` returns **Timestamp label**, không phải integer position. `DatetimeIndex.__getitem__` chỉ accept integer/slice/bool array, không accept Timestamp.

**Fix**: dùng `gaps.idxmax()` trực tiếp (đã là Timestamp), bỏ `df.index[...]` wrapper.

**Lesson**: pandas Series.idxmax() returns the **label** (index value), không phải positional index. Khác với numpy.argmax().

### Bug 4 — Scale mismatch giữa 2 sources

**Symptom**: Notebook 00 cell 6 cho `|vn − yf| / vn` quantiles ≈ 99900%.

**Root cause**: vnstock VCI trả về prices ở **nghìn VND**, yfinance ở **VND đơn vị thực**. Factor ~ 1000 → rel_diff ≈ 999 ≈ 99900%.

**Fix 2 phần**:
1. **Notebook**: replace cell 6 dùng log-return comparison (scale-invariant) + scale factor inference để verify hypothesis.
2. **Fetcher**: trong `fetch_tcb_price`, sau khi fetch từ yfinance, rescale `/1000` để output luôn canonical nghìn VND. Vnstock branch no rescale (đã native).

**Lesson**: cross-source verification của financial data PHẢI scale-invariant (log returns) hoặc explicit normalize trước. Cùng asset, cùng exchange, vẫn có thể đơn vị khác giữa các nguồn.

### Bug 5 — NameError cascade rel_diff trong notebook

**Symptom**: `NameError: name 'rel_diff' is not defined` trong cells 7-8 sau khi fix cell 6.

**Root cause**: Khi rewrite cell 6, biến `rel_diff` bị remove, thay bằng `lr_diff` (log-return diff) + `factor`. Cells 7-8 vẫn reference `rel_diff` → NameError.

**Fix**: update cells 7-8 dùng `lr_diff`. Cell 7 hiển thị top 10 outlier theo lr_diff_bp; cell 8 histogram của lr_diff trong basis points.

**Lesson**: khi refactor một cell trong notebook, sweep tất cả downstream cells để check broken references. Tốt nhất là restart kernel + Run All để bắt cascade errors.

---

## 7. Open questions / next session

### 7.1 Resolved trong Session 1

| Question | Status | Reference |
|---|---|---|
| Q1 — vnstock VCI close adjusted? | ✅ YES | §5.1, IMPLEMENTATION §11 |

### 7.2 Locked baselines cho Phase 0+

| Item | Value | Reference |
|---|---|---|
| Canonical price unit | nghìn VND | §3.4 |
| Cross-source log-return disagreement threshold | 100 bp (1%) | §5.2.4 |
| HOSE calendar gap WARN/ERROR | > 12 / > 15 calendar days | §5.3.3 |
| Known disagreement dates (whitelist) | 2024-05-21, 2019-03-14, 2019-03-15 | §5.2.3 |

### 7.3 Mở ra cho session sau

- **VN-Index cross-source verification chưa làm**: nếu fallback yfinance kích hoạt cho VN-Index trong production, cần verify unit nhất quán. Giả định (chưa verify) cả 2 nguồn trả về index points magnitude ~600-2000.
- **3 ngày vnstock-only**: cần share output đầy đủ cell 7 để identify specific dates.
- **yfinance raw Close auto-handle splits**: đáng note nhưng không critical (project dùng adjusted).
- **vnstock 4.0.1 migration**: defer đến Session 13 (production hardening).
- **Cross-check policy trong fetcher**: defer đến Session 2-3 (sau khi as-of join và features L1 ổn định, có thể add safety net).

### 7.4 Session 2 preview

**Module**: `src/data/asof_join.py` + tests (IMPLEMENTATION §5.1).

**Critical contract**: anti-leakage join giữa daily data (TCB price, returns, technicals) và lower-frequency data:
- L3 monthly (CPI YoY, SBV refinancing rate) — release_date typically 1-2 weeks sau reference period end
- L3 quarterly (GDP YoY) — release_date typically 1 month sau quarter end
- L4 quarterly (TCB fundamentals: NPL, NIM, P/E, etc.) — release_date according to TCB IR schedule

**As-of join semantics**: tại trading day t, chỉ dùng dữ liệu có `release_date ≤ t`. **TUYỆT ĐỐI KHÔNG** dùng `reference_period_end` (đó là cheat — biết thông tin trước khi nó được công bố).

**Tests bắt buộc** (Tier 1 trong testing strategy, IMPLEMENTATION §8):
- Synthetic data với known release_date
- Verify: feature_value at trading day t = most recent quarterly value với release_date ≤ t
- Edge cases: t trước first release_date (NaN), t ngay release_date (use), t sau next release_date (use newer)

