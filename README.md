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
- Conda (Anaconda hoặc Miniconda) — recommended cho project có CUDA
- Python 3.11 (trong conda env `ds`)
- Git
- GPU NVIDIA + CUDA driver (cho LSTM training; CPU fallback hoạt động nhưng chậm)

### Cài đặt

```powershell
git clone https://github.com/HuuKhanh19/Data-Science.git
cd Data-Science
```

#### Option A — Clone base env (nếu base có Python 3.11.x + torch CUDA)

```powershell
conda activate base
python --version    # confirm 3.11.x

conda create -n ds --clone base -y
conda activate ds
pip install -r requirements.txt
```

#### Option B — Fresh env

```powershell
conda create -n ds python=3.11 -y
conda activate ds
# Install PyTorch với CUDA — adjust cuda version theo driver
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

Verify CUDA hoạt động:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
```

### Kiểm tra setup

```powershell
conda activate ds
python -c "import numpy, pandas, sklearn, torch, vnstock, yfinance; print('OK')"
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
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
