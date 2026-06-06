# TCB Direction — Dự đoán hướng biến động tích lũy giá cổ phiếu Techcombank

> Công cụ nghiên cứu đánh giá **predictability** của giá cổ phiếu Techcombank (TCB, HOSE)
> ở đa horizon. **KHÔNG phải khuyến nghị hay lời khuyên đầu tư.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6)
![Status](https://img.shields.io/badge/Status-Research%20tool-8957e5)
![Model](https://img.shields.io/badge/Model-frozen%20(no%20refit)-d8a657)

Bài toán: phân loại nhị phân trên chuỗi thời gian — dự đoán hướng biến động tích lũy của
giá đóng cửa **điều chỉnh** sau `k ∈ {1, 5, 10, 20}` phiên giao dịch:

```
y_{t,k} = sign(P_{t+k} − P_t),   quy ước hòa (P_{t+k} = P_t) → +1
```

Dự án theo **Framework A** (kỹ thuật/dự đoán): mục tiêu là một mô hình tốt nhất kèm **một
con số hiệu năng out-of-sample trung thực**, không phải tuyên bố khoa học hình thức. Mô
hình được **huấn luyện một lần rồi đóng băng** (không refit); các phase sau chỉ chạy suy
luận (inference) trên dữ liệu mới.

---

## Mục lục

- [1. Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
- [2. Các gói phần mềm sử dụng](#2-các-gói-phần-mềm-sử-dụng)
- [3. Cài đặt môi trường](#3-cài-đặt-môi-trường)
- [4. Cấu trúc mã nguồn](#4-cấu-trúc-mã-nguồn)
- [5. Chạy chương trình](#5-chạy-chương-trình)
  - [5.1 Phase 1 — pipeline tĩnh (huấn luyện → đánh giá → web local)](#51-phase-1--pipeline-tĩnh-huấn-luyện--đánh-giá--web-local)
  - [5.2 Phase 2 — cập nhật inference real-time](#52-phase-2--cập-nhật-inference-real-time)
  - [5.3 Phase 3 — xuất bản công khai](#53-phase-3--xuất-bản-công-khai)
  - [5.4 Tự động hóa theo lịch](#54-tự-động-hóa-theo-lịch)
- [6. Đầu ra (artifacts)](#6-đầu-ra-artifacts)
- [7. Nguồn dữ liệu](#7-nguồn-dữ-liệu)
- [8. Ghi chú & giới hạn](#8-ghi-chú--giới-hạn)

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
| :--- | :--- |
| Hệ điều hành | Windows (đã kiểm thử trên Windows Server); Linux/macOS chạy được phần CPU |
| Python | **3.11** |
| Trình quản lý môi trường | Miniconda / Anaconda (khuyến nghị), hoặc `venv` |
| RAM | ≥ 8 GB |
| GPU (tùy chọn) | NVIDIA CUDA — **chỉ cần cho mô hình LSTM** (horizon `k=5`). Elastic Net và LightGBM chạy hoàn toàn trên CPU. Đã kiểm thử trên 2× RTX 5070 Ti 16GB, CUDA 12.9 |
| Git | cần cho Phase 3 (xuất bản GitHub Pages) |

> **Lưu ý GPU (RTX 50xx / kiến trúc Blackwell, `sm_120`)**: bản PyTorch ổn định mặc định có
> thể báo `torch.cuda.is_available() = True` nhưng **thiếu kernel** cho `sm_120`, khiến LSTM
> lỗi runtime. Cần cài bản PyTorch dựng theo **CUDA ≥ 12.8 (cu128)** hỗ trợ `sm_120`. Nếu
> không có GPU phù hợp, vẫn chạy được toàn bộ pipeline bằng cách bỏ LSTM (xem §5.1).

---

## 2. Các gói phần mềm sử dụng

Toàn bộ phụ thuộc được liệt kê trong [`requirements.txt`](requirements.txt). Bảng dưới mô
tả vai trò từng nhóm:

| Gói (pip) | import | Vai trò trong dự án |
| :--- | :--- | :--- |
| `pandas` | `pandas` | Khung dữ liệu, đọc/ghi Parquet, as-of join, xử lý chuỗi thời gian |
| `numpy` | `numpy` | Tính toán mảng, vector hóa metric |
| `pyarrow` | `pyarrow` | Backend đọc/ghi `.parquet` |
| `scikit-learn` | `sklearn` | Elastic Net logistic (`LogisticRegression`), `StandardScaler`, các metric (balanced-acc, MCC, AUC, …) |
| `lightgbm` | `lightgbm` | Mô hình cây phi tuyến (GBDT) — `LGBMClassifier` |
| `torch` (PyTorch) | `torch` | Mô hình LSTM (deep). **Cài bản CUDA phù hợp GPU** (xem §1, §3) |
| `matplotlib` | `matplotlib` | Vẽ sơ đồ học (learning curves), figures đánh giá |
| `joblib` | `joblib` | Lưu/nạp artifact mô hình EN & LightGBM đã đóng băng |
| `vnstock` | `vnstock` | Lấy giá TCB & VNIndex (adjusted close) và báo cáo tài chính TCB (nguồn VCI) |
| `yfinance` | `yfinance` | Tỷ giá USD/VND (`USDVND=X`) |
| `sdmx1` | `sdmx` | Chỉ số CPI Việt Nam qua IMF Data Portal (SDMX) — **lưu ý tên cài là `sdmx1`, tên import là `sdmx`** |
| `requests`, `beautifulsoup4`, `lxml` | `requests`, `bs4` | Lấy GDP từ VBMA (TSV gốc GSO) |
| `jupyterlab` (hoặc `notebook`) | — | Chạy notebook EDA train-only (Step 8) |

> Phần web (dashboard) là **HTML tĩnh + ECharts nạp qua CDN**, không cần gói server nặng.
> Web local được phục vụ bằng HTTP server tĩnh của Python (không cần FastAPI/Flask).

---

## 3. Cài đặt môi trường

### Bước 1 — Clone repository

```bash
git clone https://github.com/HuuKhanh19/Data-Science.git
cd Data-Science
```

### Bước 2 — Tạo môi trường conda `ds`

```bash
conda create -n ds python=3.11 -y
conda activate ds
```

### Bước 3 — Cài PyTorch theo GPU (làm TRƯỚC `requirements.txt`)

Chọn đúng build CUDA cho card của bạn. Ví dụ cho GPU Blackwell (RTX 50xx, `sm_120`):

```bash
# Bản hỗ trợ CUDA 12.8 (cu128) — kiểm tra hướng dẫn chính thức tại pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Nếu **không dùng GPU** (chạy CPU, bỏ LSTM): có thể cài bản CPU `pip install torch`, hoặc
bỏ qua hẳn `torch` — pipeline tự động chạy được EN + LightGBM mà không cần PyTorch.

### Bước 4 — Cài các phụ thuộc còn lại

```bash
pip install -r requirements.txt
```

### Bước 5 — Kiểm tra nhanh GPU (tùy chọn)

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> Trên Windows, các runner đã chủ động `import torch` **trước** `numpy/pandas/lightgbm` để
> tránh xung đột thứ tự nạp DLL (MKL). Bạn không cần làm gì thêm.

---

## 4. Cấu trúc mã nguồn

```
Data-Science/
├── data/
│   ├── raw/                  # 6 nguồn .parquet (tái tạo bằng fetch_phase1.py; .gitignore)
│   ├── interim/              # trung gian (labels.parquet, …)
│   ├── processed/            # features.parquet, predictions_baseline/model.parquet
│   └── forward_log.parquet   # Phase 2 — sổ track record live (append-only)
├── src/
│   ├── data/                 # schema, validation, _common, fetch_*, clean, integrate,
│   │                         #   features, label, assemble, forward_log
│   ├── model/                # split, baselines, tuning, train_infer, evaluate
│   └── web/                  # inference (ráp dữ liệu cho dashboard)
├── scripts/                  # runner mỏng *_phase1.py + phase2_update.py +
│                             #   publish_phase3.py + run_phase2.bat
├── config/                   # feature_sets.json, hparams.json, deploy_choice.json
├── deploy/                   # manifest.json + artifact mô hình đã đóng băng (*.joblib/*.pt)
├── reports/                  # results.json, figures/, eda/
├── notebook/                 # eda_train_only.ipynb (Step 8)
├── web/                      # index.html + app_data.json (bản dev) + tcb_dashboard.html
├── docs/                     # index.html (bản xuất bản GitHub Pages)
├── logs/                     # phase2.log
├── data.md                   # TÀI LIỆU DỮ LIỆU CORE (đọc trước khi sửa pipeline)
├── requirements.txt
└── README.md
```

Mỗi bước theo **kiến trúc lai**: logic thuần trong `src/**` (DataFrame in → DataFrame out,
không đụng đĩa) + một runner mỏng `scripts/*_phase1.py` (file-in/file-out, trả mã thoát
`0/1`, ghi log `_*_log.json`). Nhờ vậy logic dễ kiểm thử và Phase 2 chỉ việc bọc lại.

---

## 5. Chạy chương trình

> Tất cả lệnh chạy **từ thư mục gốc repo**, trong môi trường `conda activate ds`.

### 5.1 Phase 1 — pipeline tĩnh (huấn luyện → đánh giá → web local)

Chạy lần lượt theo đúng thứ tự (mỗi bước đọc đầu ra của bước trước):

```bash
# Step 1 — Thu thập & đóng băng 6 nguồn raw (snapshot END_DATE = 2026-05-29)
python scripts/fetch_phase1.py

# Step 2 — Làm sạch + dựng trục lịch HOSE (spine)
python scripts/clean_phase1.py

# Step 3 — Tích hợp (merge daily + as-of join theo release_date)
python scripts/integrate_phase1.py

# Step 4 — Sinh 20 đặc trưng ứng viên (toàn bộ lag ≤ t−1, chống rò rỉ)
python scripts/features_phase1.py

# Step 5 — Gán nhãn y_{t,k} cho k ∈ {1,5,10,20}
python scripts/label_phase1.py

# Step 6 — Lắp ráp → data/processed/features.parquet (cắt warmup)
python scripts/assemble_phase1.py
```

**Step 7 (chia tập 80:10:10 + embargo `k`)** không có runner riêng — được gọi nội tuyến
bởi các bước mô hình thông qua `src/model/split.py`.

**Step 8 — EDA train-only** (sinh `config/feature_sets.json` gồm các bộ `l1` / `eda` / `full`):

```bash
jupyter lab    # mở và chạy notebook/eda_train_only.ipynb
```

Tiếp tục phần mô hình hóa & đánh giá:

```bash
# Step 9 — Baselines (persistence / dynamic-majority / always-positive)
python scripts/baselines_phase1.py

# Step 10 — Tune hyperparameter trên tập val (EN: λ; LightGBM: grid; LSTM: frozen-config)
python scripts/tune_phase1.py

# Step 11 — Huấn luyện + suy luận (chạm test 1 lần) + CHỌN deploy + đóng băng deploy/
python scripts/train_infer_phase1.py
#   Bỏ LSTM (không cần GPU):
python scripts/train_infer_phase1.py --models elastic_net lightgbm
#   Chỉ định card GPU cho LSTM:
python scripts/train_infer_phase1.py --gpu 1

# Step 12 — Đánh giá test (acc/balanced-acc/MCC/AUC + confusion + cờ degenerate)
python scripts/evaluate_phase1.py
```

**Step 13 — Web local:**

```bash
# Ráp dữ liệu cho dashboard (web/app_data.json) và bản dashboard tự chứa
python scripts/web_build_phase1.py
python scripts/web_app_phase1.py     # sinh web/tcb_dashboard.html (mở trực tiếp được)

# Phục vụ bản dev (web/index.html cần HTTP server vì fetch không chạy với file://)
python scripts/web_serve_phase1.py
```

`web/tcb_dashboard.html` là deliverable **tự chứa** — mở trực tiếp trên mọi máy không cần
server.

### 5.2 Phase 2 — cập nhật inference real-time

Mô hình **đóng băng, KHÔNG refit**. Phase 2 chỉ làm mới dữ liệu, ghi/chấm dự đoán forward
và dựng lại web:

```bash
python scripts/phase2_update.py             # fetch → tầng DATA → cập nhật sổ forward → rebuild web
python scripts/phase2_update.py --skip-fetch   # dùng lại data/raw hiện có (không gọi mạng)
python scripts/phase2_update.py --no-open       # không tự mở trình duyệt (chạy headless)
```

Phase 2 **không** chạy bước chọn/đánh giá mô hình: `predictions_model.parquet` và
`reports/results.json` giữ nguyên (bất biến). Bằng chứng tích lũy mới nằm ở
`data/forward_log.parquet` — mỗi dự đoán forward được tự chấm đúng/sai khi đủ `k` phiên.

### 5.3 Phase 3 — xuất bản công khai

Sao chép dashboard tĩnh sang `docs/` rồi commit & push lên GitHub Pages
(Settings → Pages → Deploy from branch: `main`, thư mục `/docs`):

```bash
python scripts/publish_phase3.py            # copy → docs/index.html, commit & push (chỉ khi có thay đổi)
python scripts/publish_phase3.py --no-push   # commit nhưng không push (để test)
```

### 5.4 Tự động hóa theo lịch

`scripts/run_phase2.bat` gắn vào **Windows Task Scheduler** (khuyến nghị 04:00 giờ VN): chạy
`phase2_update.py`, và chỉ khi thành công (`exit 0`) mới chạy tiếp `publish_phase3.py`. Nhật
ký ghi vào `logs/phase2.log`. Chỉnh các biến `REPO`, `ENVDIR` trong file `.bat` cho khớp máy
của bạn trước khi tạo task.

---

## 6. Đầu ra (artifacts)

| Đường dẫn | Nội dung |
| :--- | :--- |
| `data/processed/features.parquet` | Panel daily 1742 × (date + 20 đặc trưng thô + 4 nhãn) |
| `data/processed/predictions_baseline.parquet` | Dự đoán của các baseline |
| `data/processed/predictions_model.parquet` | Dự đoán OOS của lưới (feature_set × model × k) |
| `config/hparams.json`, `config/deploy_choice.json` | Hyperparameter đã khóa & cấu hình deploy đã chọn |
| `deploy/manifest.json` + `deploy/*.joblib`/`*.pt` | Mô hình đã refit-toàn-bộ và **đóng băng** (best-single mỗi `k`) |
| `reports/results.json` | Kết quả test (1 lần) toàn lưới + tóm tắt |
| `reports/figures/`, `reports/eda/` | Sơ đồ học, biểu đồ đánh giá, quyết định EDA |
| `web/tcb_dashboard.html` | Dashboard tự chứa (deliverable) |
| `data/forward_log.parquet` | Sổ track record live (Phase 2) |

> `data/raw/*` và artifact mô hình (`*.joblib`, `*.pt`) **không** được commit (`.gitignore`);
> chúng được tái tạo bằng `fetch_phase1.py` và `train_infer_phase1.py`.

---

## 7. Nguồn dữ liệu

| Yếu tố | Nguồn | Tần suất |
| :--- | :--- | :--- |
| Giá TCB (adjusted close), VNIndex | vnstock (VCI) | Daily |
| Tỷ giá USD/VND | yfinance (`USDVND=X`) | Daily |
| CPI Việt Nam | IMF Data Portal — SDMX `VNM.CPI._T.IX.M` (qua `sdmx1`) | Monthly |
| Tăng trưởng GDP | VBMA (TSV gốc GSO) | Quarterly |
| Cơ bản TCB (tổng TS, P/E, NPL, dư nợ, NIM, vốn CSH) | vnstock VCI (báo cáo tài chính) | Quarterly |

Mỗi nguồn lưu kèm `release_date` (ngày thông tin **thực sự công bố**) để as-of join chống rò
rỉ tương lai. Chi tiết đầy đủ trong [`data.md`](data.md).

---

## 8. Ghi chú & giới hạn

- **Chống rò rỉ tương lai (anti-leakage)**: đặc trưng tại phiên `t` chỉ dùng thông tin đến
  hết phiên `t−1`; macro nối bằng as-of join theo `release_date`; embargo `k` phiên ở mỗi
  ranh giới chia tập; chuẩn hóa z-score chỉ fit trên train.
- **Đánh giá**: dùng **balanced-accuracy / MCC**, không dùng accuracy thô (dữ liệu mất cân
  bằng lớp; tập val rơi vào một regime tăng mạnh). Test chỉ được chạm **một lần**.
- **Mô hình đóng băng**: không refit ở Phase 2 → có thể trôi (concept drift) khi dữ liệu
  live đi xa vùng huấn luyện. Đây là đánh đổi đã chấp nhận của dự án.
- **Tái lập GPU**: card Blackwell cần PyTorch CUDA `sm_120` (xem §1, §3), nếu không LSTM
  (`k=5`) sẽ lỗi; phần còn lại vẫn chạy trên CPU.

---

> ⚠️ **Tuyên bố miễn trừ**: Đây là **công cụ nghiên cứu** đánh giá khả năng dự đoán hướng giá
> TCB — khả năng này được tìm thấy là **rất yếu**. Kết quả **không** phải khuyến nghị hay lời
> khuyên đầu tư. Đầu tư có rủi ro; bạn tự chịu trách nhiệm với quyết định của mình.