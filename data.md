# Dữ liệu dự án: Từ bài toán → nhóm ảnh hưởng → raw → feature cho model

**Bài toán**: dự đoán hướng biến động tích lũy giá Techcombank (TCB, HOSE) ở đa horizon $k \in \{1,5,10,20\}$ phiên — phân loại nhị phân trên chuỗi thời gian.

**Cấu trúc dự án (Phase / Step)**:
- **Phase 1 (hiện tại)** — làm trên dữ liệu tĩnh. **Mỗi bước là một Step riêng**: Step 1 thu thập raw → Step 2 EDA → Step 3 làm sạch → Step 4 tích hợp → Step 5 feature → Step 6 gán nhãn → Step 7 lắp ráp (ra `features.parquet`) → Step 8 cầu nối walk-forward → Step 9 baselines → Step 10 tuning → Step 11 train+infer → Step 12 đánh giá → **Step 13 web local**. Output Phase 1: **web app chạy local**.
- **Phase 2** — tự động hóa toàn bộ quy trình.
- **Phase 3** — deploy web cho người khác truy cập.

> **Đang ở Phase 1 — đã xong Step 1→7: chuỗi xử lý dữ liệu hoàn tất, `data/processed/features.parquet` (1742 phiên × 20 feature + 4 nhãn, `2019-06-07→2026-05-28`) sẵn sàng cho model.** Tài liệu này chốt phần **dữ liệu** (Step 1→7) và **cầu nối sang mô hình** (Step 8); roadmap mô hình hóa → web local (Step 8→13) ở Section 5. Mạch logic: bài toán → khảo sát nhóm ảnh hưởng → vì sao collect raw → vì sao biến đổi thành feature → tập feature cho model → lộ trình hoàn tất Phase 1.
>
> Chi tiết từng Step xử lý dữ liệu: `docs/phase1_step{1..7}_*.md`.

---

## 1. Khảo sát: nhóm dữ liệu ảnh hưởng đến giá cổ phiếu

Feature phải **có căn cứ** chứ không chọn tùy tiện, nên bước đầu là khảo sát literature xem *nhóm yếu tố nào tác động đến giá cổ phiếu ngân hàng*. Kết quả là 3 nhóm dưới đây, về sau ánh xạ thành **Technical → L2, Vĩ mô → L3, Nội tại → L4**; thêm **L1** là log-return giá thuần làm baseline tối giản.

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

| Yếu tố ảnh hưởng | Lớp | Nguồn raw | Tần suất | File output |
| :--- | :--- | :--- | :--- | :--- |
| Giá TCB (nền của L1 + nhãn + L2) | L1/L2 | Giá đóng cửa **điều chỉnh** — vnstock (VCI) | Daily | `tcb_price.parquet` |
| Chỉ số VNIndex | L3 | vnstock (VCI) | Daily | `vnindex.parquet` |
| Tỷ giá hối đoái | L3 | yfinance `USDVND=X` | Daily | `usdvnd.parquet` |
| Lạm phát | L3 | IMF Data Portal — CPI all-items index (SDMX `VNM.CPI._T.IX.M`, qua `sdmx1`) | Monthly | `cpi.parquet` |
| Tăng trưởng GDP | L3 | VBMA TSV (gốc GSO) | Quarterly | `gdp.parquet` |
| Tổng TS, P/E, NPL, dư nợ, NIM, vốn CSH | L4 | vnstock VCI Finance (`_get_financial_report`, bảng `ratio` cho P/E) | Quarterly | `tcb_fundamentals.parquet` |

**Nguyên tắc tầng raw**: chỉ *thu thập trung thực, chưa biến đổi*. Lưu kèm `release_date` (ngày thông tin **thực sự công bố** — để chống rò rỉ tương lai) và `fetched_at`; mỗi nguồn một parquet **schema khóa cứng**. CPI lấy thẳng dạng số (chỉ số IMF), kèm `release_date` theo quy ước (cuối tháng tham chiếu + 6 ngày, do nguồn số không có ngày công bố chính thức).

**Kết quả** (`scripts/fetch_phase1.py`, tính đến 29/05/2026 — 6/6 OK): `tcb_price` 1994 dòng (2018-06-04→2026-05-28), `vnindex` 2086, `usdvnd` 2078, `cpi` 109 (chỉ số IMF, ref 2017-03→2026-03, trễ ~2 tháng so với GSO), `gdp` 37, `tcb_fundamentals` 33 (ref 2018-Q1→2026-Q1).

**Biến đã loại** (limitation): lãi suất (không nguồn time-series ổn định), số chi nhánh IBR (chỉ có theo năm), spread/lag dài (thiếu citation).

---

## 3. Chuỗi xử lý dữ liệu Phase 1 — Step 1→7 (raw → features.parquet)

Phần này biến 6 nguồn raw thành **một panel daily sẵn sàng cho model**. Trục xuyên suốt là **chống rò rỉ tương lai (anti-leakage)** — khác hẳn dữ liệu cross-section (drug, taxi) vốn được phép shuffle. Quy ước nhân quả: **đặc trưng tại phiên $t$ chỉ dùng thông tin đến hết phiên $t-1$**; chỉ **nhãn** mới được nhìn tương lai.

> **Quy ước Step**: mỗi mục "Step N" dưới đây là **một Step độc lập** của Phase 1, mỗi Step là một module logic thuần (`src/data/*.py`) + runner mỏng (`scripts/*_phase1.py`, file-in/file-out, exit 0/1) + doc (`docs/phase1_stepN_*.md`). Step 1 (thu thập raw) đã mô tả ở Section 2; dưới đây nhìn nó từ góc *đầu vào của xử lý*.

**Input**: 6 file `data/raw/*.parquet` (Section 2). **Output**: `data/processed/features.parquet`.

### Step 1 — Thu thập dữ liệu thô (Data Acquisition)
* **Đầu vào:** 6 file `data/raw/*.parquet` đã được schema-locked, bao gồm `tcb_price`, `vnindex`, `usdvnd` (khung daily OHLCV), `cpi` (monthly), `gdp` (quarterly), và `tcb_fundamentals` (quarterly).
* **Xử lý biến chậm:** Các nguồn dữ liệu trễ đã kèm theo `release_date` với quy ước bảo thủ (CPI +6 ngày, GDP +30 ngày, fundamentals +45 ngày). Bước này giữ nguyên dữ liệu thô để làm đầu vào cho các khâu sau.

### Vai trò của EDA trong bài toán này (bối cảnh cho Step 2)

> Phần này giải thích *vì sao* EDA ở bài toán này có hình hài như vậy — vì sao nó nằm đầu pipeline, vì sao nó tối giản, và khác EDA thông thường ở đâu. Kết quả thực tế của EDA xem `docs/phase1_step2_eda.md`.

**EDA có hai vai trò, bài này chỉ làm một.**
EDA (Exploratory Data Analysis), do Tukey (1977) khởi xướng, vốn là hoạt động *khám phá* — vẽ, soi, để **phát hiện** pattern và hình thành giả thuyết. Trong một pipeline thông thường, chu trình khoa học dữ liệu đặt "exploration & visualization" ở **gần cuối**, sau làm sạch và tích hợp (Slide môn học). Nhưng còn một vai trò thứ hai ít được gọi tên: **precheck** — "nhìn trước khi chạm" để đánh giá *chất lượng* dữ liệu, như bài thực hành NYC Taxi làm *trước* tiền xử lý. Step 2 của ta đảm nhiệm vai trò thứ hai này, và vì thế đặt ở đầu.

**Vì sao bài này cắt bỏ nửa "khám phá".**
Đây là nghiên cứu **confirmatory + pre-registered** (xem `research_design.md`). Pre-registration — khóa feature/nhãn/cách đánh giá *trước khi* chạm test set — là cơ chế chống "researcher degrees of freedom" và p-hacking (Simmons, Nelson & Simonsohn 2011; Nosek et al. 2018). Hệ quả: nửa "khám phá" của EDA (soi tương quan feature–nhãn, chọn lọc feature) **bị vô hiệu hóa có chủ đích** — nếu làm *trước* model, nó sẽ rò rỉ chính quan hệ ta đang muốn kiểm định vào phép kiểm định.

Nên EDA ở đây là **"xác nhận, không khám phá"**: kiểm chứng các giả định đã ghi (vd. $P_t$ non-stationary bằng ADF — Dickey & Fuller 1979; return phi-Gaussian) và đánh giá chất lượng dữ liệu. Nó chỉ nuôi các quyết định cấp *triển khai* ("how it was computed", chi tiết ở `docs/phase1_step*.md`, được sửa tự do), không đụng quyết định *thiết kế* ("what was analyzed", đã khóa).

**Vì sao precheck phải nằm đầu.**
Vì đầu ra của nó (cách căn FX, độ dài warmup, độ tin của adjusted close, mức imbalance) là *đầu vào* cho làm sạch và tích hợp. Nguyên tắc "đầu vào là rác thì đầu ra là rác" (Slide môn học) đặt khâu đánh giá chất lượng lên trước mọi phép biến đổi.

**"EDA khám phá" không biến mất — nó dời ra sau model.**
Nửa khám phá quan hệ vẫn còn, nhưng bị đẩy *qua khỏi* model thành (1) trực quan kết quả (accuracy theo horizon, DM/bootstrap CI) và (2) interpretability bằng SHAP (Phase 6). Đặt sau model để không bẻ cong phép kiểm định.

**Vì sao "tiết chế" lại là "chặt chẽ".**
Đa số nghiên cứu ML dự đoán chứng khoán làm rất nhiều EDA và feature selection *trước* model, và đó thường chính là nguồn leakage/overfit khiến kết quả không tái lập (đối lập với weak-form EMH của Fama 1970, vốn dự báo return khó dự đoán). Ngược lại, thực hành dự báo nghiêm túc (Diebold & Mariano 1995) tiết chế snooping tiền-model và đánh giá out-of-sample một cách hình thức — đúng dòng thiết kế này theo (xem thêm Pham et al. 2016; Nguyen et al. 2023 cho bối cảnh thị trường Việt Nam).

**Trích dẫn:**
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Dickey, D. A. & Fuller, W. A. (1979). Distribution of the Estimators for Autoregressive Time Series with a Unit Root. *JASA*, 74(366).
- Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2).
- Simmons, J. P., Nelson, L. D. & Simonsohn, U. (2011). False-Positive Psychology. *Psychological Science*, 22(11).
- Nosek, B. A. et al. (2018). The preregistration revolution. *PNAS*, 115(11).
- Diebold, F. X. & Mariano, R. S. (1995). Comparing Predictive Accuracy. *JBES*, 13(3).
- Pham et al. (2016); Nguyen et al. (2023) — bối cảnh thị trường Việt Nam, dẫn trong `research_design.md` §1.4.

### Step 2 — Khám phá & kiểm định sơ bộ (EDA / Phase 0)
* **Kiểm định giá điều chỉnh:** Quan sát chuỗi giá TCB, xác nhận không có các đứt gãy (jump) bất thường không thể giải thích, đặc biệt quanh sự kiện chia cổ tức 1:1 năm 2024.
* **Kiểm định tính dừng (ADF Test):** Kiểm tra $P_t$ (kỳ vọng non-stationary để loại bỏ giá thô) và log-return (kỳ vọng stationary).
* **Khảo sát phân phối & lệch lịch:** Ước lượng class imbalance cho mỗi horizon $k$, đếm số lượng ties $P_{t+k} = P_t$ (quy ước gán $+1$) và ghi nhận các điểm lệch lịch/missing data giữa các nguồn khác nhau. Bước này chỉ quan sát, tuyệt đối chưa sửa dữ liệu.

### Step 3 — Làm sạch dữ liệu (Data Cleaning) & Dựng trục lịch HOSE
* **Dựng trục thời gian chuẩn (Date Spine):** Sử dụng tập ngày giao dịch thực tế của `tcb_price` (lịch HOSE). Tuyệt đối không tự sinh business-day để tránh bị lệch ngày nghỉ lễ thực tế tại Việt Nam.
* **Làm sạch dữ liệu giá:** Đối với TCB và VNINDEX, xác nhận ngày tăng nghiêm ngặt, `close > 0`. Không forward-fill giá hoặc return nếu phiên đó không giao dịch.
* **Căn chỉnh tỷ giá & Vĩ mô:** Reindex USD/VND về lịch HOSE, forward-fill mức tỷ giá cho các ngày HOSE giao dịch mà FX bị thiếu. Các dữ liệu CPI/GDP/fundamentals được giữ nguyên như đã validate ở Step 1.

### Step 4 — Tích hợp dữ liệu (Data Integration)
* **Daily Merge:** Ghép nối trực tiếp VNINDEX và USD/VND vào trục HOSE theo `date` (cùng tần suất, độ an toàn cao).
* **Lõi chống leakage (As-of Join):** Đối với L3 (CPI/GDP) và L4 (fundamentals), sử dụng as-of join backward theo điều kiện `release_date` $\le t$ (không dùng `reference_period`). Chốt chặn này đảm bảo tại phiên $t$, mô hình chỉ nhìn thấy số liệu đã thực sự công bố.
* **Forward-fill sau As-of Join:** Giữ nguyên giá trị của lần công bố gần nhất cho các khoảng trống giữa hai kỳ release để mô phỏng chính xác trạng thái "biết gì dùng nấy" trên production.

### Step 5 — Biến đổi & Kỹ thuật đặc trưng (Feature Engineering)
* **Tính toán 20 feature nhân quả:** Tạo L1 (log-return các khung thời gian), L2 (các chỉ báo kỹ thuật MA, Momentum, Bollinger, RSI, MACD), L3 (vĩ mô YoY/% change) và L4 (cơ bản YoY). Tất cả feature tại $t$ chỉ dùng dữ liệu tính đến $P_{t-1}$.
* **Hoãn chuẩn hóa:** Cố tình không thực hiện Z-score tại bước này. Việc chuẩn hóa được hoãn sang Step 8 (cầu nối) để tránh rò rỉ dữ liệu thống kê từ tương lai vào quá khứ.

### Step 6 — Gán nhãn (Labeling)
* **Tính toán Target:** Sử dụng công thức $y_{t,k} = \mathrm{sign}(P_{t+k} - P_t)$ với $k \in \{1, 5, 10, 20\}$. Đây là bước duy nhất được phép nhìn vào tương lai.
* **Quy ước ties & Xử lý đuôi:** Nếu $P_{t+k} = P_t$, quy ước gán $y = +1$. Đối với $k$ phiên cuối cùng của mỗi horizon, gán giá trị NaN (vẫn giữ dòng để phục vụ inference live, nhưng không đưa vào tập train).

### Step 7 — Lắp ráp, xử lý NA & Xuất Artifact
* **Cắt vùng warmup:** Loại bỏ phần đầu chuỗi bị NaN do độ trễ của các indicator (momentum cần khoảng 252 phiên). Vùng dữ liệu khả dụng bắt đầu từ giữa năm 2019 (trùng khớp thời điểm L4 YoY thu thập đủ 4 quý).
* **Không giảm chiều dữ liệu:** Không dùng PCA và không feature selection. Tập 20 feature đã được pre-registered; việc chọn lọc thêm sẽ biến thành p-hacking và phá vỡ tính confirmatory.
* **Kiểm định đầu ra:** Đảm bảo file `data/processed/features.parquet` không còn NaN trong vùng khả dụng (trừ nhãn đuôi) và khóa `date` không bị trùng lặp.

### Step 8 — (Cầu nối) Walk-Forward split & Z-score per-window

> **Đây là cầu nối, KHÔNG phải một Step xử lý dữ liệu tĩnh như Step 1→7.** Step 1→7 mỗi cái nuốt file vào, nhả một artifact bền ra. Step 8 thì **không ghi file tĩnh** vì hai lý do bản chất: (a) walk-forward tạo **~150–208 cặp train/test** (mỗi tuần refit một cặp), không có "tập đã chia" duy nhất để lưu; (b) `μ,σ` của Z-score **phải fit riêng trên train window từng tuần** — chuẩn hóa sẵn thành một file tĩnh sẽ dùng `μ,σ` toàn cục (có cả tương lai) → rò rỉ. Vì thế Step 8 là **module logic thuần** (`src/model/walk_forward.py`: splitter generator + scaler fit-trên-train) chạy *bên trong* vòng lặp refit (Step 11), biến `features.parquet` thành ma trận per-window phù du trong RAM. Không `*_phase1.py` ghi parquet, không `_*_log.json` kiểu Step 1→7 — thay vào đó là unit test trên dữ liệu giả lập.

* **Chia tập Non-Random:** rolling window 1000 phiên, refit hàng tuần, có **buffer gap $k$ phiên** ở cuối train để loại nhãn chưa thực sự quan sát được (chốt chặn leakage 3/4).
* **Chuẩn hóa Z-score an toàn:** fit $\mu,\sigma$ **chỉ trên train window** mỗi tuần rồi mới apply vào test tuần đó (chốt chặn leakage 4/4).
* **⚠ Cần đối chiếu:** mốc Phase-0 window / test split trong `research_design.md §3.1` (2018-06-04→2022-06-30) được đặt trên spine đầy đủ 1994 phiên *trước khi* chốt warmup. Vùng khả dụng thực bắt đầu **2019-06-07** (1742 phiên), nên ranh giới walk-forward phải dời theo — giải quyết khi build splitter ở Step 8.

**Bốn chốt chặn leakage**: (1) feature tính lag (≤ $t-1$, Step 5); (2) as-of join theo `release_date` (Step 4); (3) walk-forward có buffer gap $k$ (Step 8); (4) Z-score fit chỉ trên train (Step 8). Z-score và phân chia tập **không** nằm trong `features.parquet` mà ở vòng lặp refit của Step 11.

---

### EDA Phase 0 — đính chính & quy tắc khóa (29/05/2026)

Sau EDA Phase 0 (`notebooks/eda_phase0.ipynb`), chốt các quy tắc xử lý áp dụng từ Step 3 (làm sạch) trở đi:

- **Adjusted close**: sạch (0 phiên |log-return|>15%; biến động tuân trần ±7% HOSE) → dùng thẳng.
- **FX (USD/VND)**: thiếu 239 ngày HOSE → reindex về lịch HOSE + forward-fill mức tỷ giá, tính %change sau.
- **Ties nhãn**: tie-rate `k=1` ≈ 7.8% (do giá tick rời rạc) → giữ quy ước tie→+1, flag tie-rate (xem `research_design.md` §2.2). Số ties toàn chuỗi: 155/46/21/11 cho `k`=1/5/10/20.
- **L4 thiếu NPL/NIM 2 quý**: 2018-Q1 (trước niêm yết + warmup) → drop tự nhiên; 2021-Q2 → forward-fill NPL & NIM từ 2021-Q1 (không leakage; không recompute được do mẫu số thiếu); `interest_earning_assets` (33/33 NaN) → bỏ, NIM lấy thẳng `nim_pct`.
- **Warmup**: cắt vùng đầu chuỗi ở Step 7 → vùng khả dụng thực **2019-06-07** (cắt 252 phiên, mốc do `momentum_3_12` chi phối).

## 4. Tập feature sẵn sàng cho model

**Schema `data/processed/features.parquet`**: `date` + **20 feature** + **4 nhãn**. Feature ở dạng **thô (chưa chuẩn hóa)**. Công thức đầy đủ + citation: `research_design.md` Section 4.

| Lớp | Số | Feature |
| :--- | :--- | :--- |
| **L1** — giá thuần (daily) | 4 | $r_{t-1}=\log(P_{t-1}/P_{t-2})$; cumulative 5/10/20 phiên $\log(P_{t-1}/P_{t-1-h})$ |
| **L2** — kỹ thuật (daily) | 6 | MA crossover $\mathrm{MA}_5/\mathrm{MA}_{20}$; Momentum 3-12mo (skip 63 phiên); Bollinger position; Trading Range Breakout $\in\{-1,0,1\}$; RSI(14); MACD chuẩn hóa (chia $P_{t-1}$) |
| **L3** — vĩ mô (mixed, ffill từ `release_date`) | 4 | VN-Index return; USD/VND % change; CPI YoY; GDP YoY |
| **L4** — cơ bản TCB (quarterly, ffill từ `release_date`) | 6 | Total Assets growth YoY; P/E; NPL ratio; Credit growth YoY; NIM; Equity/Assets |

**Nhãn**: $y_{t,1}, y_{t,5}, y_{t,10}, y_{t,20} \in \{-1,+1\}$ — mỗi horizon là một bài phân loại nhị phân độc lập.

**Lưu ý dùng tập này**: chuẩn hóa Z-score và phân chia walk-forward thực hiện ở Step 8+11 (per-window), không bake vào file — đảm bảo không một con số tương lai nào lọt vào quá khứ.

---

## 5. Lộ trình Step 8→13 — hoàn tất Phase 1 (đích: web local)

`features.parquet` đã đóng chuỗi *dữ liệu*. Phần còn lại của Phase 1 là **mô hình hóa → đánh giá → web local**. Mỗi Step giữ kiến trúc lai (logic thuần `src/model/*.py` + runner mỏng `scripts/*_phase1.py`, exit 0/1, `_*_log.json`), trừ Step 8 là module infra (không artifact). Mô hình & baseline đã pre-registered trong `research_design.md` (§3, §5, §6, §7).

| Step | Tên | Input → Output (artifact) | Vai trò chính |
| :--- | :--- | :--- | :--- |
| **8** | Cầu nối walk-forward + Z-score | `features.parquet` → *(không file; module RAM)* | Splitter rolling 1000 + buffer gap `k`; scaler fit-trên-train. Đóng chốt leakage 3+4 |
| **9** | Baselines | `features.parquet` → `data/processed/predictions_baseline.parquet` | Persistence, dynamic-majority, always-+1 — đặt **rào** để DM test so |
| **10** | Hyperparameter tuning (Phase-0) | `features.parquet` (chỉ Phase-0 window) → `config/hparams.json` | Tune Elastic Net / LightGBM / LSTM **một lần**, khóa lại. Không chạm test (chống p-hacking) |
| **11** | Walk-forward train + inference | `features.parquet` + `hparams.json` → `data/processed/predictions_model.parquet` | Step nặng (GPU LSTM). ~150–208 tuần refit × 3 model × 4 horizon, log 0-1 loss |
| **12** | Đánh giá thống kê | `predictions_*.parquet` → `reports/results.json` + `reports/figures/*` | Accuracy + 95% CI (block bootstrap, block `2k`); BalAcc/MCC; **DM vs baselines** (HAC lag `k-1`); đường **predictability theo `k`** |
| **13** | **Web local** (deliverable Phase 1) | `predictions_*` + `results.json` → app chạy `localhost` | (a) inference hôm nay: 4 dự đoán `k∈{1,5,10,20}` cho phiên mới nhất; (b) trực quan kết quả khoa học (predictability-vs-`k`, DM, confusion, calibration) |

**Thứ tự phụ thuộc**: 8 (infra, mọi thứ cần) → 9 & 10 (độc lập nhau, đều cần 8) → 11 (cần 8+10) → 12 (cần 9+11) → 13 (cần 11+12).

**Ba điểm khoa học phải giữ khi sang phần model**:
1. **Rào baseline cao và tăng theo `k`**: `pct_pos` vùng khả dụng = 55.6/56.1/57.2/59.2% (`k`=1/5/10/20). Majority-class baseline ~56–59% — model phải vượt **qua DM test**, không nhìn accuracy thô. Nghịch lý: rào cao nhất đúng ở `k=20`, chỗ giả thuyết cho tín hiệu yếu nhất.
2. **Overlapping labels** (`k>1`): chuỗi nhãn tự tương quan → DM dùng HAC variance lag `q=k-1`, block bootstrap mean block `2k`.
3. **Đối chiếu walk-forward window** với vùng khả dụng 1742 phiên (xem ⚠ ở Step 8) trước khi chạy Step 11.

> **Phase 2** (sau Phase 1): tự động hóa fetch→features→refit→web theo lịch. **Phase 3**: deploy web public. Step 1→7 đã thiết kế sẵn cho Phase 2 (file-in/file-out, exit 0/1, schema-locked).

---