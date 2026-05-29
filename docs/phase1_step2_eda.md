# Phase 1 / Step 2 — EDA & Precheck (Phase 0): Kết quả

**Ngày**: 29/05/2026
**Notebook**: `notebooks/eda_phase0.ipynb`
**Phạm vi**: read-only trên `data/raw/` (6 nguồn); chỉ sinh hình tại `reports/eda_phase0/figures/`. Không tạo/sửa dữ liệu.
**Vai trò của bước**: xem `data.md` (mục "Vai trò của EDA trong bài toán này"). Tài liệu này chỉ trình bày *kết quả*.

---

## Tóm tắt

5/5 kiểm tra đạt. EDA **xác nhận** các giả định thiết kế (giá điều chỉnh sạch, $P_t$ non-stationary, log-return stationary, imbalance nhẹ) và lôi ra **2 phát hiện cần xử lý**: ties ở $k=1$ ≈ 7.8% (không "gần 0" như giả định ban đầu), và L4 thiếu NPL/NIM ở 2 quý. Cả hai đã được khóa thành quy tắc minh bạch (mục cuối), không làm thay đổi tập feature/nhãn đã pre-register.

---

## Kết quả từng kiểm tra

### 1. Tính toàn vẹn của adjusted close — đạt

- 1994 phiên, khoảng `2018-06-04 → 2026-05-28`; cả 3 validator (monotonic dates, calendar gap, abnormal returns) đều 0 cảnh báo, 0 lỗi.
- **0 phiên** |log-return| > 15%. Top-5 biến động đều ≈ −0.073 = $\log(0.93)$ — đúng **trần biên độ ±7%/phiên của HOSE** (phiên sàn tháng 03/2020, 10/2022, 04/2025). Đây là sự kiện thị trường thật, không phải artifact điều chỉnh.
- Hình `01`: chuỗi liền mạch; zoom 2024 không có bậc thang quanh chia cổ tức cổ phiếu 1:1 → adjustment đã apply đúng.
- **Quyết định**: dùng giá thẳng cho L1/L2/nhãn.

### 2. Phân phối nhãn (Phase-0 window) — imbalance nhẹ

| $k$ | n | %(+1) | ties |
|---|---|---|---|
| 1 | 1020 | 54.4% | 81 |
| 5 | 1016 | 52.8% | 26 |
| 10 | 1011 | 51.1% | 9 |
| 20 | 1001 | 51.0% | 5 |

- %(+1) giảm dần về ~50% khi $k$ tăng vì Phase-0 window gần như đi ngang về net (hình `01`: ~16 → đỉnh 26 → đáy 7 → ~16). %(+1) ở $k=1$ cao hơn là hiệu ứng vi cấu trúc (phiên tăng nhiều hơn nhưng biên độ nhỏ hơn).
- Cột %(+1) chính là **rào của baseline majority-class** mà model phải vượt (hình `03`).
- **Quyết định**: chấp nhận imbalance tự nhiên, **không SMOTE**; báo cáo Balanced Accuracy và MCC bên cạnh accuracy.

### 3. Tính dừng (ADF, Phase-0) — đúng kỳ vọng

- $P_t$: ADF = −0.963, p = 0.767 → **non-stationary**.
- log-return: ADF = −29.5, p < 0.001 → **stationary**.
- **Quyết định**: xác nhận **loại $P_t$ thô khỏi L1**, dùng log-return làm cơ sở. Đây là xác nhận giả định đã ghi, không phải phát hiện mới.

### 4. Ties $P_{t+k}=P_t$ — PHÁT HIỆN

- Tie-rate toàn chuỗi: $k=1$ **7.8%** (155 dòng), $k=5$ 2.3%, $k=10$ 1.1%, $k=20$ 0.6%.
- KHÔNG "gần 0" như giả định ban đầu — vì giá đóng cửa điều chỉnh **rời rạc theo bước giá (tick)**, không liên tục.
- Ở $k=1$, strict-up chỉ 46.5%; con số 54.4% phần lớn do dồn 81 ties vào +1.
- **Quyết định**: giữ quy ước tie→+1, vì (i) DM test so trên *chênh lệch* loss với cùng một nhãn nên kết luận **bất biến** với quy ước; (ii) quy ước làm baseline majority-class mạnh lên (ở $k=1$, "luôn +1" đạt 54.4%) → **nâng rào** cho model, bảo thủ. Đính chính giả định trong `research_design.md` §2.2 và **báo cáo tie-rate kèm accuracy** ở các $k$ nhỏ.

### 5. Phân phối return (descriptive) — fat tail bị trần ±7% cắt cụt

- mean = 0.00036, std = 0.0207, skew = −0.31, excess kurtosis = 2.56.
- Hình `02`: phân phối nhọn, lệch trái nhẹ (giảm nhanh hơn tăng); hai đuôi **dồn cục ở ±7%** (trần sàn HOSE). Kurtosis chỉ 2.6 — thấp so với 5–10 thường thấy ở cổ phiếu — vì trần ±7% cắt cụt đuôi.
- **Ý nghĩa method**: củng cố hai lựa chọn thiết kế — (a) dự đoán **hướng** chứ không **biên độ** (biên độ fat-tail khó model, hướng bền hơn); (b) dùng **suy luận phân-phối-tự-do** (Diebold-Mariano + block bootstrap) thay vì test giả định phân phối chuẩn.

### 6. Volatility regimes (descriptive)

- Hình `02` (phải): clustering rõ — đỉnh volatility 2020 (COVID), 2022 (siết lãi suất / stress ngân hàng) ~0.7 annualized; 2024 êm, 2025 nhảy lại.
- **Ý nghĩa method**: củng cố dùng **rolling window** (để model bám regime) và xác nhận RQ1.3 (robustness theo regime) là câu hỏi có thật. Lưu ý: Phase-0 (tuning) rơi vào vùng high-vol, test period êm hơn — ghi nhận như một đặc điểm chuyển-regime, không phải lỗi.

### 7. Lệch lịch & missing

- **FX (USD/VND)**: 2078 phiên vs 1994 phiên HOSE; **239 ngày HOSE thiếu FX**, 323 ngày FX có mà HOSE nghỉ. → Bước 3: reindex FX về lịch HOSE + forward-fill **mức** tỷ giá (tính %change sau).
- **Nguồn chậm**: CPI n=109 (lag 6d), GDP n=37 (lag 30d), fundamentals n=33 (lag 45d) — lag cố định theo quy ước Step 1. Coverage lùi về 2017 đủ cho YoY (hình `04`).
- **L4 NaN**: `interest_earning_assets` 33/33; `npl_ratio_pct` 2; `nim_pct` 2.

---

## Phát hiện & quy tắc khóa

### A. Ties (→ `research_design.md` §2.2)

Giữ quy ước tie→+1; đính chính giả định "xác suất gần 0" bằng tie-rate thực (k=1: 7.8%); báo cáo tie-rate kèm accuracy ở $k$ nhỏ. Lý do giữ quy ước: bất biến với DM test + làm baseline mạnh lên (bảo thủ).

### B. L4 fundamentals (→ `data.md`)

- **2018-Q1 (2018-03-31)**: trước ngày niêm yết + nằm trong vùng warmup → **drop tự nhiên ở Bước 7**, không cần xử lý. (Vẫn dùng làm mẫu số YoY cho Total Assets growth ở 2019-Q1 — cột TA không thiếu.)
- **2021-Q2 (2021-06-30)**: hố giữa chuỗi, active qua as-of join khoảng `2021-08-14 → 2021-11-14` (~65 phiên trong vùng khả dụng) → **forward-fill NPL & NIM từ 2021-Q1**. An toàn leakage (Q1 công bố ~05/2021, biết trước cửa sổ Q2); không recompute được vì mẫu số đều thiếu (`interest_earning_assets` toàn NaN, numerator nợ xấu không có).
- **`interest_earning_assets`** (33/33 NaN, vnstock không cung cấp): cột chết → bỏ; NIM lấy thẳng từ `nim_pct`.

---

## Tham số khóa cho Bước 3–4

| Hạng mục | Quyết định |
|---|---|
| Giá | Dùng thẳng (close sạch, 0 abnormal) |
| Date spine | Lịch HOSE = tập ngày của `tcb_price` (1994 phiên) |
| FX | Reindex về spine + forward-fill mức tỷ giá (239 ngày) |
| L3/L4 | As-of join theo `release_date` + forward-fill giữa release |
| L4 thiếu | 2018-Q1 drop (warmup); 2021-Q2 ffill từ Q1; bỏ `interest_earning_assets` |
| Warmup | Cắt vùng đầu chuỗi (~tới giữa 2019) ở Bước 7 |
| Nhãn | Chấp nhận imbalance, không SMOTE; tie→+1 + flag tie-rate |

---

## Kết luận

EDA Phase 0 hoàn tất, không có rào cản ngăn sang Bước 3 (làm sạch + dựng spine HOSE). Các giả định thiết kế được xác nhận; hai điểm lệch so với giả định ban đầu (ties, L4 missing) đã thành quy tắc minh bạch ở giai đoạn pre-lock — đúng tinh thần pre-registration: ghi nhận và đính chính, không thay đổi tập feature/nhãn.