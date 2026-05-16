# Data Science — Multi-horizon Direction Prediction for Techcombank (TCB)

**Academic project** đánh giá khả năng dự đoán hướng biến động tích lũy giá cổ phiếu Techcombank (TCB, HOSE) ở 4 horizon `k ∈ {1, 5, 10, 20}` phiên giao dịch.

- **Bài toán**: Binary classification time series, label `y_{t,k} = sign(P_{t+k} - P_t)`.
- **Mục đích khoa học**: đánh giá predictability TCB từ thông tin công khai. Bao gồm cả khả năng kết quả negative.
- **Mục tiêu**: web app live + scientific report.
- **Tác giả**: Đặng Hữu Khanh

---

## Disclaimer

Đây là project nghiên cứu khoa học. **Không phải đầu tư khuyến nghị**. Tác giả không chịu trách nhiệm cho bất kỳ quyết định tài chính nào dựa trên output của hệ thống này.

---

## Documentation

Hai tài liệu thiết kế nằm ở root repo:

- **`research_design.md`** — *Pre-registered* methodology (WHAT was analyzed). Lock cuối tuần 2, version-tagged trên Git, snapshot trên OSF. Không sửa sau lock trừ khi bug fix với version bump.
- **`IMPLEMENTATION.md`** — Code architecture (HOW it was computed). Working document, sửa free trong suốt project.

Đọc cả hai trước khi contribute.

---

## Setup

### Yêu cầu
- Python 3.11 (pin trong `pyproject.toml`)
- Git
- Khuyến nghị: GPU NVIDIA + CUDA cho LSTM training (deterministic mode, dùng được CPU nhưng chậm)

### Cài đặt

```bash
git clone https://github.com/HuuKhanh19/Data-Science.git
cd Data-Science

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Cài đặt PyTorch với CUDA (Windows + CUDA 12.x)

`requirements.txt` pin `torch>=2.4` từ PyPI (CPU). Nếu cần CUDA build, cài riêng theo hướng dẫn chính thức:

```bash
# Ví dụ cho CUDA 12.4 (backward-compatible với driver 12.9):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA hoạt động:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
```

### Kiểm tra setup

```bash
python -c "import numpy, pandas, sklearn, torch, vnstock, yfinance; print('OK')"
pytest tests/  # all tests should pass (chưa có test ở thời điểm bootstrap)
```

---

## Folder structure

Xem chi tiết trong `IMPLEMENTATION.md` section 3. Tóm tắt:

```
data-science-tcb/
├── research_design.md     # PRE-REGISTERED, locked post-Phase 0
├── IMPLEMENTATION.md      # Working architectural doc
├── CHANGELOG.md
├── requirements.txt
├── pyproject.toml
├── data/                  # raw + processed + sqlite (gitignored)
├── src/                   # Python modules
│   ├── data/              # fetchers, asof_join
│   ├── features/          # L1-L4 feature layers + pipeline
│   ├── models/            # ElasticNet, RandomForest, LSTM (BaseModel)
│   ├── eval/              # walk_forward, metrics, baselines, dm_test, bootstrap
│   ├── interpret/         # SHAP + permutation
│   ├── ablation/
│   ├── storage/           # SQLite ORM
│   ├── app/               # Streamlit
│   ├── inference/         # daily.py, weekly_refit.py
│   └── utils/             # config, logging, seeds
├── notebooks/             # EDA + Phase 0 + verification
├── scripts/               # One-shot runners (acquire, backtest, ablation, shap)
└── tests/                 # pytest
```

---

## Reproducibility

- **Seed**: `42` đặt ở mọi entry point qua `src/utils/seeds.set_global_seed()`.
- **Pinned versions**: `requirements.txt` lock khi end of Phase 0.
- **Git tags**: `v1.0-prereg-locked` cuối tuần 2.
- **OSF snapshot**: immutable archive của repo state tại lock.

---

## License

MIT — xem `LICENSE`.
