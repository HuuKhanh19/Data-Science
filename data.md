# Dữ liệu dự án: Từ bài toán → nhóm ảnh hưởng → raw → feature cho model

**Bài toán**: dự đoán hướng biến động tích lũy giá Techcombank (TCB, HOSE) ở đa horizon $k \in \{1,5,10,20\}$ phiên — phân loại nhị phân trên chuỗi thời gian.

**Khung tiếp cận**: **kỹ thuật/dự đoán (Framework A)** — mục tiêu là một model dự đoán tốt nhất kèm **một con số performance out-of-sample trung thực**, không phải một tuyên bố khoa học. Không pre-registration, không kiểm định giả thuyết hình thức (Diebold-Mariano/Holm/bootstrap/verdict). Cho phép feature selection (quyết trên train, validate trên val) và chọn/ensemble model tự do.

**Cấu trúc dự án (Phase / Step)**:
- **Phase 1 (hiện tại)** — làm trên dữ liệu tĩnh. Trình tự 13 Step (xem dưới). Output: **web app chạy local**.
- **Phase 2** — tự động hóa fetch→features→**inference** theo lịch. **KHÔNG refit**: model train một lần ở Phase 1 rồi đóng băng; Phase 2 chỉ chạy suy luận trên dữ liệu mới.
- **Phase 3** — deploy web public.

**Trình tự xử lý (Framework A)** — lưu ý EDA nằm **sau** lắp ráp + chia tập (không phải ở đầu):

```
1 Raw → 2 Clean (+integrity precheck) → 3 Integrate → 4 Features (20 ứng viên)
→ 5 Label → 6 Assemble (features.parquet) → 7 Split 80:10:10 + embargo k
→ 8 EDA (TRAIN-ONLY) → 9 Baselines → 10 Tune+Select (val)
→ 11 Train+Infer (test 1 lần; deploy refit-all) → 12 Evaluate → 13 Web (inference-only)
```

> **Trạng thái**: Step 1 đã done — 6 nguồn `data/raw/*.parquet` đóng băng snapshot `END_DATE = 2026-05-29` (phiên HOSE cuối 2026-05-28). Rebuild Step 2→13 theo Framework A. Tài liệu này là **tham chiếu dữ liệu CORE** của dự án.

---

## 1. Khảo sát: nhóm dữ liệu ảnh hưởng đến giá cổ phiếu

Feature phải **có căn cứ** chứ không chọn tùy tiện, nên bước đầu là khảo sát literature xem *nhóm yếu tố nào tác động đến giá cổ phiếu ngân hàng*. Kết quả là 3 nhóm dưới đây, về sau ánh xạ thành **Technical → L2, Vĩ mô → L3, Nội tại → L4**; thêm **L1** là log-return giá thuần làm baseline tối giản. (Đây là *pool ứng viên*; việc giữ/bỏ feature nào quyết ở Step 8 EDA train-only + Step 10 val.)

### Bảng 1: Nhóm dữ liệu Vĩ mô và Nội tại

| Nhóm dữ liệu | Nguồn dữ liệu cụ thể | Tác động | Trích dẫn |
| :--- | :--- | :--- | :--- |
| Vĩ mô | Tốc độ tăng trưởng GDP | Cùng chiều (+) | Giá cổ phiếu... bị ảnh hưởng bởi các yếu tố gồm: Tăng trưởng GDP... có mối quan hệ thuận chiều sti.vista.gov |
| Vĩ mô | Chỉ số VNIndex | Cùng chiều (+) | Kết quả cho thấy rằng chỉ số VNIndex và tăng trưởng GDP có ảnh hưởng đến giá CP ngân hàng digital.lib.ueh.edu |
| Vĩ mô | Tỷ giá hối đoái (EX) | Cùng chiều (+) | Các yếu tố vĩ mô như tốc độ tăng trưởng (GDP) và tỷ giá hối đoái (EX) có ý nghĩa thống kê và tác động cùng chiều lên biến TGCP (dlib.hvtc.edu) |
| Vĩ mô | Lạm phát (INF) | Ngược chiều (−) | Tỷ lệ lạm phát (INF) tác động nghịch chiều lên PRI (dlib.hvtc.edu) |
| Nội tại | Quy mô ngân hàng (Tổng tài sản) | Cùng chiều (+) | Các biến vi mô: tổng tài sản... đều có ý nghĩa thống kê (digital.lib.ueh.edu); quy mô (ASS)... tác động cùng chiều lên TGCP dlib.hvtc.edu |
| Nội tại | Hệ số P/E (giá/trên-thu nhập) | Cùng chiều (+) | Hệ số giá trên thu nhập (PE)... tác động cùng chiều lên TGCP (dlib.hvtc.edu) |
| Nội tại | Tỷ lệ nợ xấu | Ngược chiều (−) | Tổng nợ xấu đều có ý nghĩa thống kê, khẳng định vai trò quan trọng của đặc điểm nội tại ngân hàng (digital.lib.ueh.edu) |
| Nội tại | Tăng trưởng tín dụng | Phức tạp (phân hóa) | Tăng trưởng tín dụng không còn là yếu tố quyết định duy nhất. Thị trường ngày càng tập trung vào chất lượng tăng trưởng, khả năng kiểm soát NIM (tuoitre) |
| Nội tại | NIM (biên lợi nhuận ròng) | Ngược chiều (−) | Lợi nhuận biên... có mối quan hệ ngược chiều (sti.vista.gov) |
| Nội tại | Tỷ lệ vốn chủ sở hữu/tổng tài sản | Cùng chiều (+) | Tỷ lệ vốn chủ sở hữu trên tổng tài sản... có mối quan hệ thuận chiều (sti.vista.gov) |

### Bảng 2: Nhóm dữ liệu Phân tích Kỹ thuật (Technical features)

| Nhóm dữ liệu | Nguồn dữ liệu cụ thể | Tác động | Trích dẫn |
| :--- | :--- | :--- | :--- |
| Technical feature | Moving Average Crossover (SMA/EMA, ví dụ 1-50, 1-150) | Cùng chiều (+) | Trading rules dựa trên moving averages tạo lợi nhuận significant trên Dow Jones 1897-1986, không thể giải thích bằng random walk, AR(1), hay GARCH-M onlinelibrary.wiley.com |
| Technical feature | Past return momentum (return 3-12 tháng) | Cùng chiều (+) | Chiến lược mua winners bán losers dựa trên past returns 3-12 tháng tạo positive returns significant; thách thức weak-form EMH onlinelibrary.wiley.com |
| Technical feature | Bollinger Bands position | Phức tạp | Bollinger Bands breakout method có predictive power significant trên 14 thị trường quốc tế, return trung bình 0.294% so với market return 0.026% sciencedirect.com |
| Technical feature | Trading Range Breakout (vượt mức cao/thấp gần đây) | Cùng chiều (+) | Trading Range Breakout rule tạo lợi nhuận vượt baseline trên DJIA, returns sau buy signals cao hơn và ít biến động hơn sell signals onlinelibrary.wiley.com |
| Technical feature | RSI (Relative Strength Index, 14-period) | Phức tạp | RSI với ngưỡng 60-40 cho return tốt hơn ngưỡng truyền thống 70-30 trên NIFTY 50 (Ấn Độ); evidence trên emerging markets mạnh hơn developed markets papers.ssrn.com |
| Technical feature | MACD (Moving Average Convergence Divergence) | Cùng chiều (+) | MACD optimized với historical volatility cho prediction accuracy cao hơn 33% so với MACD truyền thống cho stock price prediction onlinelibrary.wiley.com |

---

## 2. Vì sao collect các nguồn raw này (Step 1 — đã xong)

Mỗi yếu tố ở Phần 1 cần **một nguồn đo được, tự động hóa được** cho 2018–2026. Map yếu tố → nguồn:

| Yếu tố ảnh hưởng | Lớp | Nguồn raw (cơ chế) | Tần suất | File output |
| :--- | :--- | :--- | :--- | :--- |
| Giá TCB (nền của L1 + nhãn + L2) | L1/L2 | Giá đóng cửa **điều chỉnh** — vnstock (VCI) | Daily | `tcb_price.parquet` |
| Chỉ số VNIndex | L3 | vnstock (VCI) | Daily | `vnindex.parquet` |
| Tỷ giá hối đoái | L3 | yfinance `USDVND=X` | Daily | `usdvnd.parquet` |
| Lạm phát | L3 | IMF Data Portal — CPI all-items index (SDMX `VNM.CPI._T.IX.M`, qua `sdmx1`) | Monthly | `cpi.parquet` |
| Tăng trưởng GDP | L3 | VBMA TSV (gốc GSO, web scrape) | Quarterly | `gdp.parquet` |
| Tổng TS, P/E, NPL, dư nợ, NIM, vốn CSH | L4 | vnstock VCIFinance (`_get_financial_report`, method private) | Quarterly | `tcb_fundamentals.parquet` |

Code Step 1 (refactor theo nguồn): `src/data/_common.py` + `fetch_prices.py` (tcb_price, vnindex) + `fetch_fx.py` (usdvnd) + `fetch_cpi.py` + `fetch_gdp.py` + `fetch_fundamentals.py`; orchestrator `scripts/fetch_phase1.py`.

**Nguyên tắc tầng raw**: chỉ *thu thập trung thực, chưa biến đổi*. Lưu kèm `release_date` (ngày thông tin **thực sự công bố** — chống rò rỉ tương lai) và `fetched_at`; mỗi nguồn một parquet **schema khóa cứng**. `END_DATE` khóa cứng = `2026-05-29` (snapshot tĩnh Phase 1; Phase 2 mới fetch động).

**Kết quả** (snapshot 29/05/2026 — 6/6 OK): `tcb_price` 1994 dòng (2018-06-04→2026-05-28), `vnindex` 2086, `usdvnd` 2078, `cpi` 109 (ref 2017-03→2026-03), `gdp` 37, `tcb_fundamentals` 33 (ref 2018-Q1→2026-Q1).

**Biến đã loại** (limitation): lãi suất (không nguồn time-series ổn định), số chi nhánh IBR (chỉ có theo năm), spread/lag dài (thiếu citation).

**Rủi ro nguồn cho Phase 2**: giá (vnstock) + FX (yfinance) + CPI (IMF SDMX) là API chuẩn, bền. Hai điểm dễ vỡ cần monitor: **GDP scrape VBMA** (đổi layout/encoding) và **fundamentals gọi method private VCIFinance** (vnstock đổi internal).

---

## 3. Chuỗi xử lý Step 1→7 — raw → features.parquet → split

Biến 6 nguồn raw thành **một panel daily sẵn sàng cho model** rồi chia tập. Trục xuyên suốt là **chống rò rỉ tương lai (anti-leakage)** — khác hẳn dữ liệu cross-section (drug, taxi) vốn được phép shuffle. Quy ước nhân quả: **đặc trưng tại phiên $t$ chỉ dùng thông tin đến hết phiên $t-1$**; chỉ **nhãn** mới được nhìn tương lai.

> Mỗi Step là module logic thuần (`src/data/*.py`) + runner mỏng (`scripts/*_phase1.py`, file-in/file-out, exit 0/1).

**Input**: 6 file `data/raw/*.parquet`. **Output**: `data/processed/features.parquet` + chỉ số split.

### Step 1 — Thu thập raw (đã xong)
6 parquet schema-locked (Section 2). Biến chậm kèm `release_date` (CPI +6 ngày, GDP +30 ngày, fundamentals +45 ngày).

### Step 2 — Làm sạch + Precheck chất lượng (gộp integrity vào đây)
* **Precheck integrity** (trên toàn data — chỉ kiểm tra tính toàn vẹn, không đụng quyết định feature–nhãn nên không leak): missing, dtype, range, gap lịch FX, sanity adjusted-close.
* **Dựng trục thời gian (Date Spine)**: dùng tập ngày giao dịch thực tế của `tcb_price` (lịch HOSE). Không tự sinh business-day.
* **Làm sạch giá**: TCB & VNINDEX xác nhận ngày tăng nghiêm ngặt, `close > 0`; không forward-fill giá/return phiên không giao dịch.
* **Căn chỉnh FX**: reindex USD/VND về lịch HOSE, forward-fill mức tỷ giá cho ngày HOSE thiếu FX. CPI/GDP/fundamentals giữ như validate ở Step 1.

### Step 3 — Tích hợp (Data Integration)
* **Daily Merge**: ghép VNINDEX và USD/VND vào trục HOSE theo `date`.
* **Lõi chống leakage (As-of Join)**: với L3 (CPI/GDP) và L4 (fundamentals), as-of join backward theo `release_date $\le t$` (KHÔNG dùng `reference_period`) → tại phiên $t$ chỉ thấy số liệu đã thực sự công bố. *(Chốt leakage 2/4.)*
* **Forward-fill sau as-of**: giữ giá trị lần công bố gần nhất cho khoảng trống giữa hai kỳ release.

### Step 4 — Feature engineering (20 feature ứng viên)
* Tính L1 (log-return các khung), L2 (MA, Momentum, Bollinger, Trading Range Breakout, RSI, MACD), L3 (vĩ mô YoY/%change), L4 (cơ bản YoY). **Mọi feature tại $t$ chỉ dùng dữ liệu đến $P_{t-1}$.** *(Chốt leakage 1/4.)*
* **Hoãn chuẩn hóa**: không Z-score ở đây; chuẩn hóa fit-trên-train ở thời điểm modeling (Step 11) để tránh rò rỉ thống kê từ tương lai.
* 20 feature là **pool ứng viên**, không cố định — selection quyết ở Step 8 (train-only) + Step 10 (val).

### Step 5 — Gán nhãn (Labeling)
* $y_{t,k} = \mathrm{sign}(P_{t+k} - P_t)$, $k \in \{1,5,10,20\}$. Đây là bước **duy nhất** được nhìn tương lai.
* Ties ($P_{t+k}=P_t$) → $+1$. $k$ phiên cuối mỗi horizon → NaN (giữ dòng cho inference, loại khỏi train).

### Step 6 — Lắp ráp & Xuất artifact
* **Cắt warmup**: bỏ phần đầu chuỗi NaN do độ trễ indicator (`momentum_3_12` ~252 phiên) → vùng khả dụng từ **2019-06-07** (1742 phiên).
* **Kiểm định**: `features.parquet` không còn NaN trong vùng khả dụng (trừ nhãn đuôi); khóa `date` không trùng.
* Output: `data/processed/features.parquet` = `date` + 20 feature ứng viên (thô) + 4 nhãn.

### Step 7 — Chia tập 80:10:10 + embargo (định nghĩa chỉ số)
> **Thay cho walk-forward window=1000 cũ.** Bỏ refit Phase 2 nên không cần rolling window; một split thời gian cố định là đủ và đơn giản hơn.

* **Split thời gian 80:10:10** (train/val/test) **đúng thứ tự, KHÔNG shuffle**: train (80% cũ nhất) fit; val (10% giữa) tune+chọn; test (10% mới nhất) chạm **một lần**.
* **Embargo $k$ phiên ở mỗi ranh giới** (train→val, val→test): bỏ $k$ dòng cuối đoạn trước, vì nhãn overlapping nhìn $k$ phiên tới sẽ rỉ sang đoạn sau. *(Chốt leakage 3/4 — thay cho buffer gap walk-forward cũ.)*
* Step này chỉ **định nghĩa chỉ số** train/val/test. *Áp dụng* split (fit Z-score chỉ trên train rồi apply val/test — **chốt leakage 4/4**; chạm val/test) nằm ở Step 11.

**Bốn chốt chặn leakage** (là *correctness*, không phải research): (1) feature lag $\le t-1$ (Step 4); (2) as-of join theo `release_date` (Step 3); (3) embargo $k$ mỗi ranh giới split (Step 7); (4) Z-score fit chỉ trên train (Step 11). Cộng snapshot freeze `END_DATE=2026-05-29` (Step 1).

### Phát hiện chất lượng dữ liệu (đính chính áp dụng từ Step 2)
- **Adjusted close**: sạch (0 phiên |log-return|>15%; biến động tuân trần ±7% HOSE) → dùng thẳng.
- **FX (USD/VND)**: thiếu 239 ngày HOSE → reindex về lịch HOSE + forward-fill mức tỷ giá, tính %change sau.
- **Ties nhãn**: tie-rate `k=1` ≈ 7.8% (giá tick rời rạc) → giữ quy ước tie→+1. Số ties toàn chuỗi: 155/46/21/11 cho `k`=1/5/10/20.
- **L4 thiếu NPL/NIM 2 quý**: 2018-Q1 (trước niêm yết + warmup) → drop tự nhiên; 2021-Q2 → forward-fill NPL & NIM từ 2021-Q1 (không leakage); `interest_earning_assets` (33/33 NaN) → bỏ, NIM lấy thẳng `nim_pct`.

### Step 8 — EDA (TRAIN-ONLY)

> EDA (Tukey 1977) là hoạt động khám phá + đánh giá chất lượng + quyết định feature/transform. Chu trình giáo trình (Slide môn học) đặt "exploration & visualization" **sau** làm sạch+tích hợp, **trước** ML — đúng vị trí của EDA ở đây: sau khi panel đã lắp ráp (Step 6) và split đã định nghĩa (Step 7).

**Hai ràng buộc định hình vị trí:**
- **Train-only (80%)**: mọi phân tích & quyết định rút ra từ EDA chỉ dùng slice **train**. Val/test chưa bị động → quyết định feature "miễn phí" về độ chính xác, không thổi phồng con số test. → EDA phải nằm **sau split**.
- **Cần panel đã assemble**: EDA quyết định feature → cần 20 feature ứng viên đã tính. → EDA nằm **sau assemble**.

**Phân vai**: **train ĐỀ XUẤT** (EDA sinh quyết định + bộ feature ứng viên) → **val CHỌN** (đo bộ nào thật sự tốt hơn) → **test PHÁN** (chạm một lần). Quyết định feature từ EDA là *giả thuyết*; val xác nhận nó generalize hay chỉ khớp nhiễu train.

**Nội dung EDA (trên train):**
1. *Chất lượng*: missing, outlier, sanity adjusted-close, class balance `pct_pos` từng $k$.
2. *Đơn biến*: phân phối feature (return đuôi béo), ADF stationarity (Dickey-Fuller 1979: $P_t$ non-stationary, return stationary).
3. *Cấu trúc thời gian*: ACF return ($\approx 0$) vs ACF bình phương ($>0$, volatility clustering), trend/drift, seasonality.
4. *Đa biến*: multicollinearity heatmap (quan trọng cho Elastic Net), tương quan/mutual-info feature–nhãn từng $k$, feature importance (LightGBM nhanh trên train).
→ **Xuất**: bảng quyết định keep/drop/transform + các bộ feature ứng viên (vd `full-20` vs `subset`).

> EDA mô tả cho phần report (heatmap/ADF/phân phối) có thể làm trên toàn data vì không đổi model; chỉ quyết định *dẫn tới model* mới cần kỷ luật train-only.

**Trích dẫn**: Tukey (1977) *Exploratory Data Analysis*; Dickey & Fuller (1979) *JASA* 74(366); Fama (1970) *Journal of Finance* 25(2) — bối cảnh weak-form EMH (hướng giá khó dự đoán).

---

## 4. Tập feature ứng viên (panel)

**Schema `data/processed/features.parquet`**: `date` + **20 feature** + **4 nhãn**. Feature ở dạng **thô (chưa chuẩn hóa)**. Công thức đầy đủ nằm trong code Step 4 (`src/data/`).

| Lớp | Số | Feature |
| :--- | :--- | :--- |
| **L1** — giá thuần (daily) | 4 | $r_{t-1}=\log(P_{t-1}/P_{t-2})$; cumulative 5/10/20 phiên $\log(P_{t-1}/P_{t-1-h})$ |
| **L2** — kỹ thuật (daily) | 6 | MA crossover $\mathrm{MA}_5/\mathrm{MA}_{20}$; Momentum 3-12mo (skip 63 phiên); Bollinger position; Trading Range Breakout $\in\{-1,0,1\}$; RSI(14); MACD chuẩn hóa (chia $P_{t-1}$) |
| **L3** — vĩ mô (mixed, ffill từ `release_date`) | 4 | VN-Index return; USD/VND % change; CPI YoY; GDP YoY |
| **L4** — cơ bản TCB (quarterly, ffill từ `release_date`) | 6 | Total Assets growth YoY; P/E; NPL ratio; Credit growth YoY; NIM; Equity/Assets |

**Nhãn**: $y_{t,1}, y_{t,5}, y_{t,10}, y_{t,20} \in \{-1,+1\}$ — mỗi horizon là một bài phân loại nhị phân độc lập.

**Lưu ý**: Z-score và chia tập không bake vào file — Z-score fit-trên-train ở Step 11, split định nghĩa ở Step 7. 20 feature là *ứng viên*; bộ dùng thật do Step 8+10 quyết.

---

## 5. Lộ trình Step 9→13 — hoàn tất Phase 1 (đích: web local)

`features.parquet` + split đã đóng phần *dữ liệu*. Phần còn lại là **mô hình hóa → đánh giá → web local**. Mỗi Step giữ kiến trúc lai (logic thuần `src/model/*.py` + runner mỏng `scripts/*_phase1.py`, exit 0/1). Model ứng viên: **Elastic Net logistic** (tuyến tính), **LightGBM** (tree phi tuyến), **LSTM** (deep) — chọn/ensemble tự do trên val.

| Step | Tên | Input → Output | Vai trò |
| :--- | :--- | :--- | :--- |
| **9** | Baselines | `features.parquet` + split → `predictions_baseline.parquet` | persistence / always-pos / dynamic-majority — **rào tham chiếu** chấm trên val/test |
| **10** | Tune + Select (val) | `features.parquet` + split → `config/hparams.json` + cấu hình chốt | tune hyper + chọn model + chọn feature-set, **TẤT CẢ trên val** |
| **11** | Train + Infer | `features.parquet` + `hparams.json` → `predictions_model.parquet` | fit train / chọn val / **test chạm 1 lần**; model deploy = **refit-on-all** đóng băng |
| **12** | Evaluate | `predictions_*` → `reports/results.json` + figures | metric test **1 lần**: accuracy, balanced-acc, MCC, (AUC) **vs baselines** |
| **13** | **Web local** | `predictions_*` + `results.json` → app `localhost` | (a) inference: dự đoán $k\in\{1,5,10,20\}$ cho phiên mới nhất; (b) trực quan performance test + so baseline |

**Thứ tự phụ thuộc**: 7 split → 8 EDA → 9 baselines & 10 tune/select (cần 7,8) → 11 (cần 8,10) → 12 (cần 9,11) → 13 (cần 11,12).

**Ba điểm kỹ thuật phải giữ:**
1. **Rào baseline cao & tăng theo `k`**: `pct_pos` = 55.6/56.1/57.2/59.2% (`k`=1/5/10/20). `always_pos` ăn ~56–59% accuracy thuần nhờ imbalance → nhìn **balanced-acc/MCC**, không chỉ accuracy thô (vốn bị base rate chi phối). Nghịch lý: rào cao nhất ở `k=20` chỗ tín hiệu kỳ vọng yếu nhất.
2. **Test là một đoạn regime liền mạch** (~8 tháng cuối); với `k=20` chỉ còn ~8–9 nhãn độc lập → ước lượng nhiễu, ghi rõ trong report. (Cái giá của việc bỏ walk-forward.)
3. **Không refit Phase 2** → model đóng băng có thể **drift** khi data live trôi xa vùng train (concept drift). Chấp nhận cho project, ghi rõ.

> **Phase 2**: tự động hóa fetch→features→**inference** (KHÔNG refit) → web. **Phase 3**: deploy public. Step 1→7 đã thiết kế sẵn cho Phase 2 (file-in/file-out, exit 0/1, schema-locked).