# Thiết kế nghiên cứu: Dự đoán hướng biến động giá cổ phiếu Techcombank (TCB) đa horizon

**Phiên bản**: v1.0-DRAFT (chưa lock chính thức)
**Ngày bắt đầu soạn thảo**: 15/05/2026
**Ngày lock dự kiến**: cuối Tuần 2 (28/05/2026), sau khi hoàn thành EDA Phase 0
**Tác giả**: Đặng Hữu Khanh
**Repository**: https://github.com/HuuKhanh19/Data-Science
**OSF DOI**: [Sẽ điền sau khi lock]
**Git tag khi lock**: `v1.0-prereg-locked`

> **Tuyên bố pre-registration**: Tài liệu này được công bố công khai trên GitHub và Open Science Framework trước khi mô hình được áp dụng lên dữ liệu kiểm tra (test set). Mọi sửa đổi sau khi lock đều phải bump phiên bản (vd. v1.1) với changelog rõ ràng giải thích lý do sửa và phần kết quả nào bị ảnh hưởng. Tài liệu này quy định **phương pháp luận** ("what was analyzed"); chi tiết triển khai mã nguồn ("how it was computed") được mô tả trong `IMPLEMENTATION.md` (không pre-registered, được chỉnh sửa tự do).

---

## 1. Câu hỏi nghiên cứu và giả thuyết

### 1.1 Câu hỏi nghiên cứu chính

Cổ phiếu **Techcombank (TCB)** niêm yết trên HOSE có tồn tại **tính khả dự đoán (predictability)** về hướng biến động tích lũy giá ở các horizon ngắn-trung hạn $k \in \{1, 5, 10, 20\}$ phiên giao dịch, dựa trên thông tin công khai (giá lịch sử, vĩ mô, cơ bản ngân hàng), trong giai đoạn nghiên cứu hay không?

### 1.2 Câu hỏi nghiên cứu phụ

1. Predictability biến đổi như thế nào theo horizon $k$?
2. Trong bốn lớp đặc trưng (L1 giá thuần / L2 kỹ thuật / L3 vĩ mô / L4 cơ bản ngân hàng), lớp nào đóng góp predictability cận biên nhiều nhất?
3. Predictability có robust qua các regime thị trường khác nhau (post-COVID, rate-hike, recovery, highs) không?

### 1.3 Giả thuyết thống kê

Cho mỗi horizon $k \in \{1, 5, 10, 20\}$ và mỗi model $m \in \{$Elastic Net, Random Forest, LSTM$\}$:

- **$H_0^{m,k}$** (null hypothesis): Mô hình $m$ ở horizon $k$ **không** có độ chính xác (accuracy) vượt persistence baseline ở mức ý nghĩa $\alpha = 0.05$ (sau hiệu chỉnh đa kiểm định Holm-Bonferroni).
- **$H_1^{m,k}$** (alternative): Mô hình $m$ ở horizon $k$ có accuracy vượt persistence baseline ở mức $\alpha = 0.05$ đã hiệu chỉnh.

**Khẳng định tổng thể** "TCB là predictable ở multi-horizon": $H_1$ được chấp nhận (predictability tồn tại) ở ≥ 2/4 horizons.

### 1.4 Đóng góp khoa học dự kiến

- **Trường hợp positive**: Bằng chứng evidence chống lại weak-form market efficiency cho TCB ở các horizon đã kiểm tra.
- **Trường hợp negative**: Bằng chứng consistent với weak-form market efficiency, bổ sung vào literature đã có cho thị trường Việt Nam (Pham et al. 2016; Nguyen et al. 2023 MDPI null result trên sentiment).
- Cả hai trường hợp đều có giá trị khoa học do dùng phương pháp luận pre-registered.

---

## 2. Dữ liệu

### 2.1 Đối tượng nghiên cứu

| Thuộc tính | Giá trị |
|---|---|
| Mã chứng khoán | TCB |
| Sàn niêm yết | HOSE (Sở Giao dịch Chứng khoán TP. Hồ Chí Minh) |
| Ngày niêm yết | 04/06/2018 |
| Biến đầu ra cơ sở | Giá đóng cửa điều chỉnh (adjusted close) $P_t$ |

### 2.2 Định nghĩa nhãn (label)

Cho phiên giao dịch $t$ và horizon $k$:

$$y_{t,k} = \mathrm{sign}(P_{t+k} - P_t) \in \{-1, +1\}$$

Trường hợp $P_{t+k} = P_t$ (xác suất gần 0 với giá liên tục): quy ước gán $y_{t,k} = +1$. Trường hợp này được flag và đếm trong báo cáo EDA.

### 2.3 Nguồn dữ liệu

| Loại | Biến | Tần suất | Nguồn primary | Nguồn fallback/cross-check |
|---|---|---|---|---|
| Giá TCB | OHLC, volume, adjusted close | Daily | vnstock library (source='VCI') | yfinance ticker `TCB.HM` |
| Chỉ số thị trường | VN-Index OHLC | Daily | vnstock | yfinance |
| Tỷ giá hối đoái | USD/VND | Daily | yfinance ticker `USDVND=X` | SBV tỷ giá tham chiếu |
| Lạm phát | CPI YoY % | Monthly | GSO (manual scrape từ gso.gov.vn) | Trading Economics |
| Lãi suất | SBV refinancing rate (lãi suất tái cấp vốn) | Monthly | SBV (sbv.gov.vn) | — |
| Tăng trưởng kinh tế | GDP YoY % | Quarterly | GSO (manual) | — |
| Cơ bản ngân hàng | 6 chỉ số (xem 4.4) | Quarterly | TCB Investor Relations (manual scrape, preserve release_date) | vnstock financial_ratio + financial_report (cross-check) |

### 2.4 Phạm vi thời gian

- **Bắt đầu**: 04/06/2018 (ngày niêm yết TCB)
- **Kết thúc**: tháng 06/2026 (deadline project + tiếp tục live đến hết 30/06/2026)
- **Tổng phiên giao dịch ước lượng**: ~2,000 phiên

### 2.5 Xử lý dữ liệu chính

- **Adjusted close**: nhà cung cấp dữ liệu xử lý corporate actions (chia tách, cổ tức tiền mặt, cổ tức cổ phiếu). EDA Phase 0 sẽ kiểm tra: (a) chuỗi giá không có jump không giải thích được, (b) monotonicity sau adjustment, (c) đối chiếu mẫu với báo cáo doanh nghiệp.
- **Holidays và phiên không giao dịch**: bỏ qua tự nhiên (chỉ giữ trading days theo lịch HOSE).
- **Missing values**: Forward-fill chỉ áp dụng cho các biến slow-changing (L3, L4) giữa các release dates. Không forward-fill giá hoặc returns.
- **As-of join** cho L3, L4: xem chi tiết Phần 4.

---

## 3. Phân chia mẫu và thiết kế Walk-Forward

### 3.1 Cấu hình walk-forward

| Thông số | Giá trị | Lý do |
|---|---|---|
| Loại | Rolling window (KHÔNG expanding) | Cho phép model adapt với regime shifts |
| Kích thước cửa sổ huấn luyện | 1000 trading days (~4 năm) | Cân bằng giữa statistical power và recency |
| Tần suất refit | Hàng tuần (weekly) | Phản ánh production deployment cadence |
| Buffer gap | $k$ ngày | Chống look-ahead bias: nhãn $y_{t,k}$ phụ thuộc $P_{t+k}$ không available tại $t$ |
| Phase 0 window | 04/06/2018 → 30/06/2022 | Window đầu tiên (~1000 phiên) cho hyperparameter tuning |
| Test period | 01/07/2022 → 30/06/2026 | ~208 weekly refits |

### 3.2 Quy trình một bước walk-forward

Cho mỗi tuần $w$ trong test period (tổng cộng ~208 weeks):

1. **Xác định cửa sổ huấn luyện**: $[T_w - 1000 - k, T_w - k]$ — bao gồm 1000 phiên rolling, có buffer gap $k$ ở cuối để loại trừ phiên có nhãn chưa quan sát được.
2. **Tính lại Z-score normalization parameters** (mean, std) **chỉ từ training window**, apply cho toàn bộ feature pipeline downstream.
3. **Refit weights** của ba models (Elastic Net, RF, LSTM) — hyperparameter cố định sau Phase 0.
4. **Dự đoán** cho 5 phiên giao dịch trong tuần $w$ (Monday → Friday).
5. **Đợi nhãn thực** $y_{t,k}$ available tại phiên $t+k$, tính 0-1 loss và log vào database.

### 3.3 Hạn chế đã thừa nhận

- Test period **không bao gồm** Q1-Q2 2022 (Jan-Jun 2022) do giới hạn dữ liệu lịch sử trước niêm yết. Period này có: SBV nâng lãi suất 4 lần (3-12/2022), VN-Index giảm ~30%. Hạn chế: kết quả primary không phản ánh predictability trong regime drawdown sâu này.
- Không có dữ liệu ngày 01-03/06/2018 (3 phiên đầu sau niêm yết) — có thể có liquidity bất thường.

### 3.4 Robustness check (không phải primary)

Sensitivity to window size: chạy thêm với window size 750 và 1250 trading days (giữ rolling, giữ test 2022-06 → present). Báo cáo trong appendix nếu primary positive; chỉ deep-dive nếu primary negative để loại trừ artifact do window size.

---

## 4. Đặc trưng (Features) — Bốn lớp

Mọi đặc trưng tại thời điểm $t$ chỉ sử dụng thông tin có thể quan sát được **trước hoặc bằng** thời điểm $t$. Đặc trưng được tổ chức thành 4 lớp tăng dần độ trừu tượng để hỗ trợ ablation analysis.

### 4.1 L1 — Lịch sử giá thuần

Chỉ bao gồm các log-returns thuần được tính từ giá đóng cửa điều chỉnh — không thêm derived statistical features (volatility, MA ratio) vốn đã thuộc về phạm vi L2 (kỹ thuật).

| Tên | Định nghĩa toán học | Đơn vị | Ghi chú |
|---|---|---|---|
| $r_{t-1}$ | $\log(P_{t-1}/P_{t-2})$ | log-return | Log-return 1 phiên gần nhất |
| $r_{t,5}$ | $\log(P_{t-1}/P_{t-6})$ | log-return | Cumulative 5 phiên |
| $r_{t,10}$ | $\log(P_{t-1}/P_{t-11})$ | log-return | Cumulative 10 phiên |
| $r_{t,20}$ | $\log(P_{t-1}/P_{t-21})$ | log-return | Cumulative 20 phiên |

**Loại trừ**: $P_t$ thô (non-stationary theo ADF test giả định, sẽ verify trong EDA). Volatility và Price/MA ratio không đưa vào L1 — chúng là derived statistics và đã được capture implicit trong L2 (BB position chứa rolling std; MA crossover capture price-to-MA relationship).

### 4.2 L2 — Đặc trưng kỹ thuật

Mỗi feature có citation methodology rõ ràng từ literature tham khảo.

| Tên | Định nghĩa | Citation |
|---|---|---|
| MA Crossover ratio | $\mathrm{MA}_5(P) / \mathrm{MA}_{20}(P)$ | Brock, Lakonishok & LeBaron (1992), Journal of Finance |
| Momentum 3-12mo | $\log(P_{t-1}/P_{t-252})$ với cắt window 63 phiên đầu | Jegadeesh & Titman (1993), Journal of Finance |
| Bollinger Band position | $\frac{P_{t-1} - \mathrm{MA}_{20}}{2\sigma_{20}}$ | Fang, Jacobsen & Qin (2014) |
| Trading Range Breakout | $\mathbb{1}[P_{t-1} > \max_{i=t-21}^{t-2}(P_i)] - \mathbb{1}[P_{t-1} < \min_{i=t-21}^{t-2}(P_i)] \in \{-1, 0, 1\}$ | Brock, Lakonishok & LeBaron (1992) |
| RSI(14) | $100 - \frac{100}{1 + RS}$, $RS = \frac{\bar{G}_{14}}{\bar{L}_{14}}$ | Panigrahi et al. (2021), NIFTY-50 |
| MACD normalized | $\frac{\mathrm{EMA}_{12}(P) - \mathrm{EMA}_{26}(P)}{P_{t-1}}$ | Wang & Kim (2018) |

**Lý do normalize MACD**: original MACD scale phụ thuộc price level (non-stationary). Chia cho $P_{t-1}$ để stationary.

### 4.3 L3 — Bối cảnh vĩ mô

5 features theo đúng danh sách nguồn dữ liệu vĩ mô đã thống nhất (file `Nhóm dữ liệu ảnh hưởng đến giá cổ phiếu.md`).

| Tên | Định nghĩa | Tần suất | Xử lý | Citation |
|---|---|---|---|---|
| VN-Index return | $r^{VNI}_{t-1} = \log(I_{t-1}/I_{t-2})$ | Daily | Direct | digital.lib.ueh.edu |
| USD/VND % daily change | $\log(X_{t-1}/X_{t-2})$ | Daily | Direct | dlib.hvtc.edu |
| Lạm phát CPI YoY % | Tỷ lệ lạm phát YoY công bố GSO | Monthly | Forward-fill từ release_date | dlib.hvtc.edu |
| Lãi suất SBV | Lãi suất tái cấp vốn SBV | Monthly | Forward-fill từ release_date | sti.vista.gov; hvnh.edu |
| GDP YoY % | Tăng trưởng GDP quý YoY | Quarterly | Forward-fill từ release_date | sti.vista.gov |

**Loại trừ**: (i) spread features $r_{TCB} - r_{VNI}$ vì không có citation từ literature tham khảo; (ii) VN-Index cumulative returns ở các lag dài hơn — giữ feature count match với 5 nguồn được tham chiếu trong literature, một feature tương ứng một nguồn.

**As-of join cho monthly/quarterly variables**: tại phiên $t$, sử dụng giá trị có `release_date` $\leq t$. Trường hợp release_date không khả dụng từ nguồn: quy ước conservative `release_date = reference_period_end + 14 ngày` (CPI), `release_date = quarter_end + 30 ngày` (GDP). Quy ước này được pre-registered và áp dụng đồng nhất.

### 4.4 L4 — Cơ bản ngân hàng TCB

6 features theo danh sách nguồn dữ liệu nội tại đã thống nhất trong file `Nhóm dữ liệu ảnh hưởng đến giá cổ phiếu.md` (đã loại IBR khỏi danh sách do hạn chế data availability — số chi nhánh chỉ public trong Annual Report yearly, gây approximation đáng kể nếu interpolate theo quarter).

| Tên | Định nghĩa | Tần suất | Nguồn | Citation |
|---|---|---|---|---|
| Total Assets growth | $(A_q - A_{q-4})/A_{q-4}$ YoY | Quarterly | TCB IR + vnstock | digital.lib.ueh.edu; dlib.hvtc.edu |
| P/E ratio | Price / EPS trailing 4 quarters | Quarterly | TCB IR | dlib.hvtc.edu |
| NPL ratio | Nợ nhóm 3-5 / Tổng dư nợ | Quarterly | TCB IR | digital.lib.ueh.edu |
| Credit growth YoY | $(L_q - L_{q-4})/L_{q-4}$ | Quarterly | TCB IR | (kết quả phân hóa theo literature) |
| NIM | Thu nhập lãi thuần / Tổng tài sản sinh lãi | Quarterly | TCB IR | sti.vista.gov; tuoitre |
| Equity/Assets ratio | Vốn chủ sở hữu / Tổng tài sản | Quarterly | TCB IR | sti.vista.gov |

**Quy tắc release_date**: TCB IR công bố báo cáo tài chính theo lịch sau quarter end. Quan sát historical pattern:
- Q1 results: release cuối tháng 4
- Q2 results: release cuối tháng 7
- Q3 results: release cuối tháng 10
- Q4/Annual: release cuối tháng 2 năm sau

Release dates cụ thể được scrape thủ công từ trang Investor Relations và lưu vào `data/tcb_fundamentals.csv` cùng với giá trị 6 metrics. Trường hợp không xác định được release_date chính xác: quy ước `release_date = reference_quarter_end + 45 ngày` (conservative).

### 4.5 Tổng kê đặc trưng

- L1: **4 features** (daily) — chỉ lag returns thuần từ giá đóng cửa
- L2: **6 features** (daily) — technical indicators
- L3: **5 features** (mixed frequency, forward-filled từ release_date)
- L4: **6 features** (quarterly, forward-filled từ release_date)
- **Tổng: 21 features** đầu vào model

---

## 5. Mô hình và protocol huấn luyện

### 5.1 Tổng quan

Ba mô hình đại diện ba mức độ phức tạp:

1. **Logistic Regression với Elastic Net regularization** (đại diện linear, interpretable)
2. **Random Forest** (đại diện tree-based ensemble, nonlinear tabular)
3. **LSTM** (đại diện deep learning, sequential)

Mỗi algorithm × horizon được huấn luyện như một **model độc lập** (4 models per algorithm × 3 algorithms = **12 models tổng**). Không multi-output, không cross-horizon parameter sharing.

### 5.2 Mô hình 1: Logistic Regression + Elastic Net

**Hàm mục tiêu** với feature vector chuẩn hóa $\mathbf{x}_t \in \mathbb{R}^{21}$ và nhãn $y_t \in \{-1, +1\}$:

$$\min_{\boldsymbol{\beta}, \beta_0} \frac{1}{N} \sum_{t=1}^{N} \log(1 + e^{-y_t(\beta_0 + \mathbf{x}_t^\top \boldsymbol{\beta})}) + \lambda \left[ \alpha \|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2} \|\boldsymbol{\beta}\|_2^2 \right]$$

**Hyperparameters (frozen sau Phase 0)**:
- $\alpha = 0.5$ (tỷ lệ trộn L1/L2)
- $\lambda$: xác định trong Phase 0 bằng 5-fold `TimeSeriesSplit` cross-validation trên window đầu tiên. Grid: `np.logspace(-4, 1, 50)`. Tiêu chí: mean validation log-loss.
- **Sensitivity check**: chạy thêm với $\lambda/2$ và $2\lambda$, báo cáo accuracy trong appendix.

**Citation**: Zou & Hastie (2005), "Regularization and variable selection via the elastic net", JRSS-B 67(2).

### 5.3 Mô hình 2: Random Forest

**Hyperparameters (frozen, không tuning)**:
- `n_estimators = 500`
- `max_depth = None` (full grown)
- `min_samples_leaf = 5`
- `max_features = 'sqrt'`
- `bootstrap = True`
- `random_state = 42`
- `criterion = 'gini'`

**Citation**: Breiman (2001), "Random Forests", Machine Learning 45(1). Defaults dựa Fernández-Delgado et al. (2014) benchmark trên 121 datasets.

### 5.4 Mô hình 3: LSTM

**Kiến trúc**:
```
Input: (batch_size, T=20, n_features=21)
    ↓
LSTM(hidden_size=32, num_layers=1, dropout=0)
    ↓
Lấy hidden state cuối: (batch_size, 32)
    ↓
Dropout(p=0.2)
    ↓
Linear(32 → 16)
    ↓
ReLU
    ↓
Linear(16 → 1)
    ↓
Sigmoid
Output: P(y=+1) ∈ (0, 1)
```

**Hyperparameters huấn luyện (frozen)**:
- Sequence length: $T = 20$ (match horizon dài nhất)
- Loss function: Binary Cross-Entropy
- Optimizer: Adam, learning rate $1 \times 10^{-3}$, default betas
- Batch size: 32
- Max epochs: 100
- Early stopping: patience = 10 epochs trên inner validation loss
- Random seed: **42 (single seed, không ensemble)**
- Device: CUDA (RTX 5070 Ti)

**Input representation cho L3, L4**: với mỗi feature slow-changing trong L3 (CPI, SBV rate, GDP) hoặc L4, giá trị tại phiên $t$ được **replicate đồng đều** qua toàn bộ T=20 timesteps của input sequence. Các feature daily (L1, L2, VN-Index, USD/VND) sử dụng giá trị thực tại từng timestep trong sequence.

**Inner validation cho early stopping**: 15% cuối training window (contiguous, time-aware split), không xáo trộn. Stats Z-score được tính trên toàn training window (bao gồm validation slice).

**Hạn chế thừa nhận**: Single seed result có thể bị ảnh hưởng bởi initialization stochasticity. Seed variance không được characterize trong báo cáo primary. Đây là design choice ưu tiên simplicity và compute economy.

### 5.5 Z-score normalization

Cho mỗi tuần $w$ refit và mỗi feature $j$:

$$\mu_j^{(w)} = \frac{1}{N_w} \sum_{t \in \text{train}_w} x_{t,j}, \quad \sigma_j^{(w)} = \sqrt{\frac{1}{N_w - 1} \sum_{t \in \text{train}_w} (x_{t,j} - \mu_j^{(w)})^2}$$

Normalization $\tilde{x}_{t,j} = (x_{t,j} - \mu_j^{(w)}) / \sigma_j^{(w)}$ áp dụng cho **mọi feature** và **mọi model**, bao gồm cả Random Forest.

**Lý do normalize cả tree-based model**:
1. Z-score là monotonic transform → không ảnh hưởng split decisions của decision trees (impurity gain bất biến với monotonic feature transform)
2. Code path consistent (một feature pipeline duy nhất cho cả 3 models) → ít bug surface, dễ test
3. Thuận lợi cho SHAP cross-model comparison (Phần 10): SHAP values trên cùng một scale của input features

**Quan trọng — chống data leakage**: Parameters $\mu_j^{(w)}, \sigma_j^{(w)}$ được tính **chỉ** từ training window của tuần $w$, không bao gồm test data hoặc tương lai. Cụ thể: parameters re-compute mỗi tuần khi refit, áp dụng cho test predictions tuần đó.

### 5.6 Class imbalance

Lựa chọn: **chấp nhận imbalance** theo natural distribution của TCB returns.

- Không SMOTE, không oversampling, không downsampling, không class weighting.
- Lý do: time series structure bị phá vỡ bởi sampling techniques (Pradeep et al. 2020 critique về SMOTE on time series).
- Class distribution per training window được báo cáo transparent trong appendix.
- Metric secondary (Balanced Accuracy, MCC) tự nhiên handle imbalance.

---

## 6. Đánh giá: metrics

### 6.1 Metric primary

**Accuracy** với 95% confidence interval:

$$\mathrm{Acc} = \frac{1}{N_{test}} \sum_{t \in \text{test}} \mathbb{1}[\hat{y}_t = y_t]$$

### 6.2 Metrics secondary

**Balanced Accuracy**:

$$\mathrm{BalAcc} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$$

**Matthews Correlation Coefficient (MCC)**:

$$\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 6.3 Metrics appendix

- Per-class precision và recall
- Confusion matrix
- Class distribution trong từng training window
- Calibration plot (reliability diagram)

---

## 7. Kiểm định thống kê

### 7.1 Diebold-Mariano test

Cho mô hình $m$ và baseline $b$ ở horizon $k$, định nghĩa loss differential:

$$d_t = L_b(t) - L_m(t), \quad L(t) = \mathbb{1}[\hat{y}_t \neq y_t]$$

Statistic DM:

$$\mathrm{DM} = \frac{\bar{d}}{\sqrt{\hat{V}_{HAC}(\bar{d})}}$$

Trong đó variance Newey-West HAC với lag truncation:

$$\hat{V}_{HAC}(\bar{d}) = \frac{1}{N}\left(\hat{\gamma}_0 + 2 \sum_{j=1}^{q} \left(1 - \frac{j}{q+1}\right) \hat{\gamma}_j\right), \quad q = k - 1$$

**Hiệu chỉnh Harvey small-sample** (Harvey, Leybourne & Newbold 1997):

$$\mathrm{DM}^* = \mathrm{DM} \cdot \sqrt{\frac{N + 1 - 2q + q(q-1)/N}{N}}$$

Reference distribution: $t_{N-1}$.

**Citations**: Diebold & Mariano (1995), JBES 13(3); Harvey, Leybourne & Newbold (1997), IJF 13(2).

### 7.2 Block Bootstrap cho 95% CI

**Phương pháp**: Stationary bootstrap của Politis & Romano (1994), JASA 89(428).

- Mean block length: $\bar{L} = 2k$ cho horizon $k$ ($\bar{L} \in \{2, 10, 20, 40\}$).
- Số bootstrap iterations: $B = 2000$.
- CI: percentile method, $[Q_{0.025}, Q_{0.975}]$ của bootstrapped accuracy distribution.

**Tham số geometric**: $p = 1/\bar{L}$ điều khiển phân phối block length.

### 7.3 Multiple testing correction

**Holm-Bonferroni** áp dụng trong **mỗi horizon** $k$ với scope:

$$M_k = 3 \text{ models} \times 3 \text{ baselines} = 9 \text{ tests}$$

P-values được sắp xếp tăng dần và adjusted theo:

$$p_{(i)}^{adj} = \min\left(1, \max_{j \leq i}\{(M_k - j + 1) \cdot p_{(j)}\}\right)$$

Reference: Holm (1979), Scandinavian Journal of Statistics.

**Không** correction giữa các horizon (đã thống nhất: kết luận horizon-specific).

### 7.4 Baselines

1. **Persistence**: $\hat{y}_t = \mathrm{sign}(P_t - P_{t-k})$. Đây là benchmark chính ("non-trivial baseline" theo Park & Irwin 2007).
2. **Dynamic majority class**: $\hat{y}_t = $ class chiếm đa số trong training window cuối cùng. Refit weekly cùng cadence với model.
3. **Analytical 50%**: thay vì simulate random predictions, kiểm định $H_0: p = 0.5$ bằng z-test một mẫu trên sample accuracy với HAC variance (loss differential = accuracy − 0.5).

---

## 8. Tiêu chí quyết định (Decision Criteria)

### 8.1 Predictability ở horizon $k$

Được claim **iff** tất cả các điều kiện sau:

1. **Ít nhất một model** (trong 3) có $p^{adj} \leq 0.05$ khi test vs **persistence baseline** (DM test với Holm-9 correction)
2. **95% CI của Δaccuracy** (model − persistence) nằm hoàn toàn trên 0
3. Kết quả robust với sensitivity check λ × {0.5, 1, 2} cho Elastic Net (cho riêng Elastic Net), và robust với window size {750, 1000, 1250} nếu kết quả primary là positive borderline.

### 8.2 Khẳng định tổng thể

- **"Strong evidence of predictability"**: positive ở ≥ 3/4 horizons
- **"Evidence of multi-horizon predictability"**: positive ở ≥ 2/4 horizons
- **"Limited evidence"**: positive ở 1/4 horizons
- **"No evidence"**: positive ở 0/4 horizons

Threshold cho khẳng định chính "TCB là predictable": **≥ 2/4 horizons**.

### 8.3 Vùng mập mờ

P-value adjusted $\in (0.05, 0.10]$: **"suggestive but inconclusive evidence"**. Báo cáo transparent, không claim positive, không claim negative.

### 8.4 Effect size minimum

Không đặt minimum effect size threshold. Báo cáo Δaccuracy với CI để reader tự đánh giá practical significance. Effect size context: với persistence baseline accuracy ước ~50-55%, Δaccuracy ≥ 2 percentage points được coi là substantial trong context direction prediction.

---

## 9. Phân tích Ablation

### 9.1 Forward ablation (primary)

Thứ tự pre-registered theo độ specific → general:

1. **L1 only**: 6 features
2. **L1 + L2**: 12 features
3. **L1 + L2 + L3**: 18 features
4. **L1 + L2 + L3 + L4**: 24 features (full)

Mỗi step test hypothesis: "Adding lớp $L_i$ có cải thiện accuracy significantly không?" qua DM test giữa step $i$ và step $i-1$.

### 9.2 Leave-one-out ablation (complementary)

Cho mỗi lớp $L_i \in \{L_1, L_2, L_3, L_4\}$, train với $\{L_1, ..., L_4\} \setminus \{L_i\}$ và so sánh với full feature set qua DM test.

### 9.3 Alternative ordering robustness

Chạy 1-2 ordering khác (vd $L_1 \rightarrow L_3 \rightarrow L_4 \rightarrow L_2$) để verify conclusion của forward ablation không phụ thuộc thứ tự pre-registered.

### 9.4 Scope ablation

Ablation chỉ chạy với **Random Forest** (model trung gian, robust, nhanh). Lý do: compute economy và avoid distract khỏi primary results.

---

## 10. Interpretability

### 10.1 Tổng quan

**Framework chính**: SHAP (SHapley Additive exPlanations) — Lundberg & Lee (2017), NeurIPS.

**Lý do**: Phương pháp duy nhất cho phép so sánh feature contribution **cross-model** trong cùng một metric (SHAP value).

### 10.2 Explainers theo model

| Model | SHAP Explainer | Native cross-check |
|---|---|---|
| Elastic Net | `LinearExplainer` (exact) | Standardized coefficients |
| Random Forest | `TreeExplainer` (exact, Lundberg et al. 2020) | Permutation importance |
| LSTM | `DeepExplainer` (approximate) | Permutation importance trên input features |

### 10.3 Scope (Level 3: full)

**Global feature importance**: mean(|SHAP|) trên toàn test set, cho mỗi (model × horizon) = 12 analyses. Aggregate cũng theo lớp $L_i$.

**Temporal evolution**: SHAP snapshots tại 4 thời điểm — cuối các năm 2022, 2023, 2024, 2025. Show drift trong feature importance theo regime.

**Local explanations**: SHAP cho từng prediction trong web app. Cho mỗi daily prediction, hiển thị top-5 contributing features với SHAP value.

---

## 11. Robustness và Sensitivity

### 11.1 Sensitivity checks báo cáo trong appendix

- **Elastic Net λ sensitivity**: chạy với $\lambda/2$ và $2\lambda$, so sánh accuracy
- **Window size robustness** (chỉ nếu primary borderline): {750, 1000, 1250} trading days
- **Alternative ablation ordering**: 1-2 ordering khác như mục 9.3
- **Loss function alternative**: Brier score thay vì 0-1 loss (appendix-only, không thay primary)

### 11.2 Class balance per regime

Báo cáo class distribution trong từng training window và từng test segment để transparent về non-stationarity.

### 11.3 Reproducibility

- Random seed cố định: 42 (cho LSTM, RF)
- Python version: 3.11.x (sẽ pin khi lock)
- Package versions: pinned trong `requirements.txt` (sẽ commit cùng pre-reg)
- Code state: git commit hash khi lock

---

## 12. Anti-patterns được tuyên bố tránh

Các practice sau **TUYỆT ĐỐI** không được áp dụng sau khi pre-registration lock:

1. Thêm features mới sau khi đã xem test results
2. Thay đổi model class hoặc thêm models mới
3. Thay đổi train/test split hoặc walk-forward protocol
4. Thay đổi statistical test config (DM lag, bootstrap parameters, Holm scope)
5. Drop "outlier" weeks hoặc subset time periods để cải thiện accuracy
6. Cherry-pick một horizon hoặc một model class cho narrative tích cực
7. Re-tune hyperparameters sau khi đã xem test results
8. Đổi loss function primary (từ 0-1) hoặc baseline canonical (persistence)

**Cơ chế ngăn chặn**: Git commit timestamp + OSF snapshot làm proof rằng `research_design.md` được lock trước khi test results được tính. Reviewer có thể verify.

**Trường hợp bug fix sau lock**: nếu phát hiện bug methodological thực sự (vd: HAC lag tính sai, off-by-one trong as-of join), phải:
1. Document bug rõ ràng trong CHANGELOG.md
2. Bump phiên bản research_design.md (v1.1, v1.2, ...)
3. Re-run toàn bộ kết quả
4. Báo cáo **cả** v1.0 (original lock) và v1.1 (fixed) results
5. Tag git mới: `v1.1-prereg-revised`

---

## 13. Reproducibility tính toán

### 13.1 Phần cứng

- CPU: AMD Ryzen Threadripper 9960X 24-Cores (24 cores, 48 logical processors)
- GPU: 2× NVIDIA GeForce RTX 5070 Ti (16GB VRAM each, 32GB total)
- Driver: 576.88, CUDA 12.9
- OS: Windows

### 13.2 Phần mềm

- Python 3.11.x
- Key packages (versions pinned trong `requirements.txt`):
  - `numpy`, `pandas`, `scipy`
  - `scikit-learn` (Elastic Net, RF, metrics, TimeSeriesSplit)
  - `torch` (LSTM)
  - `shap`
  - `statsmodels` (HAC variance)
  - `arch` hoặc custom (block bootstrap)
  - `vnstock`, `yfinance` (data sources)
  - `streamlit` (web app)
  - `sqlalchemy` + SQLite (database)

### 13.3 Random seed control

- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`
- PyTorch deterministic mode: `torch.use_deterministic_algorithms(True)` nếu compatible với CUDA setup
- sklearn: `random_state=42` mọi nơi

### 13.4 Data versioning

- Raw data snapshot lưu trong `data/raw/` với timestamp
- Processed features cache trong `data/processed/` với hash của (raw data + feature pipeline version)
- Model artifacts (weights, hyperparameters) versioned mỗi weekly refit

---

## 14. Xử lý kết quả negative

### 14.1 Triết lý

Kết quả negative — nếu không có model nào vượt persistence baseline ở ≥ 2/4 horizons — **không** là thất bại của project. Nó là một đóng góp khoa học có giá trị:

- **Consistent với weak-form market efficiency** cho TCB ở các horizon đã test (Fama 1970)
- **Bổ sung evidence** cho null result đã có trong literature Vietnam (Nguyen et al. 2023 MDPI về sentiment + Vietnamese stocks)
- **Counterbalance** một xu hướng publication bias mạnh trong ML stock prediction (positive results overrepresented)

### 14.2 Báo cáo nếu negative

- **Title** của scientific report có thể là: "Limited Evidence for Direction Predictability of TCB Stock at Multi-Horizon" hoặc tương tự — vẫn defensible.
- **Discussion section**: framing như confirmatory evidence của weak-form efficiency, so sánh với Vietnam market literature.
- **Web app**: hiển thị transparent kết quả, không claim trading utility. Disclaimer "not investment advice" giữ nguyên.

### 14.3 Hành động bị cấm sau khi nhận negative result

Tất cả các anti-patterns ở Mục 12 vẫn áp dụng. Đặc biệt KHÔNG:
- Thêm sentiment features "vì L4 không đủ"
- Switch sang horizon khác chưa pre-registered
- Đổi từ accuracy sang Brier score primary để "tìm kết quả khác"

### 14.4 Action chấp nhận sau negative result

- Thảo luận **future work**: hướng cải thiện tiềm năng (sentiment, alternative data, longer history, additional models) — đây là discussion, không phải hành động trong scope project hiện tại.
- Compare quantitatively với existing Vietnam literature.

---

## 15. Hạn chế đã thừa nhận

1. **Sample size**: TCB niêm yết 2018-06, lịch sử ngắn (~8 năm) so với US stocks (~50+ năm). Statistical power hạn chế.
2. **Single asset**: chỉ TCB, không generalize đến banking sector hoặc HOSE nói chung.
3. **Single seed LSTM**: kết quả LSTM có thể sensitive với seed initialization, không characterize variance.
4. **Test period thiếu 2022 H1**: regime rate-hike + drawdown không có trong test.
5. **Macro data quality**: GSO/SBV data có thể có revisions backward (vd CPI revised), as-of join chỉ chính xác cho release ban đầu.
6. **L4 release_date approximation**: nếu không scrape được chính xác release_date, dùng quy ước +45 ngày — có thể off bằng vài ngày.
7. **Không có sentiment features**: scope chính loại trừ news/social sentiment (đã có MDPI 2023 null result làm tham chiếu).
8. **Không có order book / intraday data**: chỉ daily closing prices.
9. **Survivorship bias zero** (TCB chưa delist) nhưng vẫn cần note trong future replication.

---

## 16. Timeline và Milestones

| Tuần | Dates | Phase | Deliverables chính |
|---|---|---|---|
| 1 | 15-21/05/2026 | Phase 0 start + Phase 1 | Setup repo + GitHub Actions skeleton; OSF account; data acquisition (giá TCB, VN-Index, USD/VND); draft research_design.md |
| 2 | 22-28/05/2026 | Phase 1 + 2 | Data acquisition L3/L4 (GSO, SBV, TCB IR manual); EDA notebook (ADF/KPSS, autocorrelation, class balance); **LOCK pre-registration** (git tag + OSF) |
| 3 | 29/05-04/06/2026 | Phase 3 + 4 | Feature engineering implementation (L1-L4 với as-of join); model implementations (EN, RF, LSTM); walk-forward engine; Phase 0 λ tuning |
| 4 | 05-11/06/2026 | Phase 4 + 5 | **Full walk-forward backtest** (12 models × 208 weeks); statistical tests pipeline (DM, bootstrap, Holm); primary results |
| 5 | 12-18/06/2026 | Phase 5 cont. | Ablation studies (forward + LOO + alt ordering); interpretability (SHAP global + temporal); sensitivity checks (λ) |
| 6 | 19-25/06/2026 | Phase 6 | Streamlit web app (3 tabs: Scientific Evidence + Live Monitoring + Interpretability); deploy HuggingFace Spaces; setup cron daily inference + weekly refit |
| 7 | 26-30/06/2026 | Phase 7 + buffer | Write scientific report; final polish; buffer cho fixes |

**Milestone gates**:
- **Cuối Tuần 2**: Pre-reg lock. Không quay lại methodology decisions sau điểm này.
- **Cuối Tuần 4**: Primary results table sẵn sàng. Decision: positive / negative / ambiguous.
- **Cuối Tuần 5**: Mọi scientific results (ablation, interpretability, sensitivity) hoàn thành.
- **Cuối Tuần 6**: Web app live, daily cron running.
- **Cuối Tuần 7**: Project delivered.

---

## Tài liệu tham khảo

### Phương pháp luận

- Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *Journal of Finance*, 47(5), 1731-1764.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383-417.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281-291.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Park, C.-H., & Irwin, S. H. (2007). What do we know about the profitability of technical analysis? *Journal of Economic Surveys*, 21(4), 786-826.
- Pesaran, M. H., & Timmermann, A. (2007). Selection of estimation window in the presence of breaks. *Journal of Econometrics*, 137(1), 134-161.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *JASA*, 89(428), 1303-1313.

### Models và features

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Fang, J., Jacobsen, B., & Qin, Y. (2014). Predictability of the simple technical trading rules: An out-of-sample test. *Review of Financial Economics*, 23(1), 30-45.
- Fernández-Delgado, M., et al. (2014). Do we need hundreds of classifiers to solve real world classification problems? *JMLR*, 15(1), 3133-3181.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.
- Panigrahi, A. K., Vachhani, K., & Sisodia, M. (2021). Trend prediction of stock prices using RSI indicator. *SSRN Working Paper*.
- Wang, J., & Kim, J. (2018). Predicting stock price trend using MACD optimized by historical volatility. *Mathematical Problems in Engineering*, 2018.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2), 301-320.

### Bối cảnh Việt Nam

- Nguyen, A. T., et al. (2023). Sentiments extracted from news and stock market reactions in Vietnam. *Information (MDPI)*, 14(4).
- Pham, T. T., et al. (2016). Stock price prediction for Vietnamese stock market based on news headlines. *IMCOM*.

---

## Changelog

- **v1.0-DRAFT** (15/05/2026): Bản nháp đầu tiên, chờ EDA findings cuối Tuần 2.
- **v1.0-LOCKED** (dự kiến 28/05/2026): Lock với git tag `v1.0-prereg-locked` và OSF snapshot.

---

*Tài liệu này được công khai trên GitHub repository [URL] với commit hash [sẽ điền], và snapshot lên Open Science Framework [DOI sẽ điền]. Mọi sửa đổi sau khi lock đều require version bump và changelog.*
