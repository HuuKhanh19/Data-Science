# Step 6 / Bước 6 (Phase 1) — Gán nhãn (Labeling)

Tài liệu này mô tả Bước 6 đã làm gì và giải thích mã nguồn. Bước 6 hiện thực khâu
**"gán nhãn"** trong pipeline Step 2 của `data.md`: từ chuỗi giá đóng cửa điều chỉnh
trên spine HOSE, tính **4 biến mục tiêu** `y_{t,k}` cho `k∈{1,5,10,20}`. Đây là
**bước DUY NHẤT trong toàn pipeline được phép nhìn vào tương lai** — hợp lệ vì nhãn
*là* cái tương lai model phải đoán, không phải feature đầu vào (20 feature đã khóa
`≤t-1` ở Bước 5, check `no_lookahead`).

---

## 1. Mục tiêu, Input, Output

| | Nội dung |
| :--- | :--- |
| **Mục tiêu** | Tính `y_{t,k}=sign(P_{t+k}−P_t)`, `k∈{1,5,10,20}`; tie→+1; đuôi `k` phiên→NaN (giữ dòng) |
| **Input** | `data/interim/integrated.parquet` (Bước 4; cột `close` = adjusted close, non-null toàn spine) |
| **Output** | `data/interim/labels.parquet` (`date` + 4 nhãn) + `data/interim/_label_log.json` |
| **Lệnh chạy** | `python scripts/label_phase1.py` (exit 0 nếu OK, 1 nếu lỗi) |
| **Kết quả** (snapshot 1994 phiên) | 1994 phiên × 4 nhãn, `2018-06-04 → 2026-05-28`, 4/4 check pass |

**Nguyên tắc thiết kế** (nối tiếp Bước 1–5):
- *Kiến trúc lai*: logic thuần (`label.py`, DataFrame in→out) + runner mỏng
  (`label_phase1.py`, file-in/file-out) → Phase 2 tự động hóa, resume/audit được.
- *Future chỉ chạm vào nhãn*: dùng đúng `close` để tính `y`; tuyệt đối không đụng tới
  20 feature → ranh giới leakage giữ nguyên.
- *Giữ dòng đuôi*: không drop `k` phiên cuối (cần cho inference live); chỉ gán NaN.
  Loại khỏi train là việc của vòng lặp refit (Step 3), không bake vào file.
- *Schema lock đầu ra*: 4 check là hợp đồng chất lượng; sai là raise.

---

## 2. Định nghĩa & quy ước

Với phiên `t` và horizon `k` (đơn vị = **phiên giao dịch**, không phải ngày lịch —
spine đã bỏ holiday nên dịch `k` index là dịch `k` phiên):

$$y_{t,k} = \mathrm{sign}(P_{t+k} - P_t) \in \{-1, +1\}$$

- **Ties** (`P_{t+k}=P_t`) → `+1`. Không hiếm: giá điều chỉnh rời rạc theo tick nên
  tie-rate toàn chuỗi là **7.8% / 2.3% / 1.1% / 0.6%** cho `k=1/5/10/20`
  (155 / 46 / 21 / 11 dòng). Giữ `+1` vì bảo thủ: nâng baseline majority-class
  (k=1, "luôn +1" = 54.4%) và DM test bất biến với quy ước (so trên *chênh lệch* loss,
  cùng nhãn). Xem `research_design.md` §2.2.
- **Đuôi** → `k` phiên cuối mỗi horizon không có `P_{t+k}` ⇒ nhãn `NaN`. Giữ nguyên dòng.

**Cài đặt** (`build_labels`): `diff = P.shift(-k) - P`; `diff≥0`→`+1` (gộp tie);
`diff<0`→`−1`; `diff` là NaN (đuôi)→giữ `NaN`. Thứ tự 2 bước `np.where` quan trọng —
NaN phải được map lại sau cùng để **không** bị gộp nhầm vào `+1`.

**dtype**: nhãn là `float64` (`{-1.0, +1.0, NaN}`) — robust cho sklearn/bootstrap
downstream; lọc train chỉ cần `.dropna()` theo cột nhãn tương ứng.

---

## 3. Quan hệ với leakage & overlapping labels

- Bước 6 **không** phải một chốt chặn leakage — nó là chỗ *duy nhất* future được phép
  vào, nhưng chỉ vào `y`. Bốn chốt chặn vẫn là: (1) lag feature `≤t-1` (Bước 5),
  (2) as-of `release_date` (Bước 4), (3) walk-forward buffer gap `k` (Bước 8),
  (4) Z-score fit trên train (Bước 8).
- **Đừng nhầm**: NaN đuôi ở đây là đuôi *toàn cục* của chuỗi. Buffer gap `k` ở cuối
  *train window* (chốt 3) là vấn đề *per-window* xử lý ở Bước 8.
- **Overlapping labels**: với `k>1`, `y_{t,k}` và `y_{t+1,k}` chồng thông tin giá ⇒
  chuỗi nhãn tự tương quan. Bước 6 chỉ *tạo ra* (bản chất bài toán); *hệ quả* được xử
  lý downstream: DM test dùng HAC variance lag `q=k−1`, block bootstrap mean block
  length `2k` (`research_design.md` §7.1–7.2).

---

## 4. Giải thích mã nguồn

```
scripts/label_phase1.py        ← runner: đọc integrated → build → validate → ghi
└── src/data/label.py          ← logic thuần
    ├── build_labels(panel)    ← §2: shift(-k), tie→+1, đuôi→NaN; ra date + 4 nhãn
    ├── tie_counts(panel)      ← đếm ties mỗi k (cho log / đối chiếu EDA)
    └── validate(labels,panel) ← §5: chạy 4 check, fail thì raise
```

- **`label.py`** thuần DataFrame, không I/O. Hằng số khóa: `HORIZONS=(1,5,10,20)`,
  `LABELS=[y_1,y_5,y_10,y_20]`, `PRICE_COL="close"`, và `_SPINE_LOCKED=1994` +
  `_TIE_COUNTS_LOCKED` (guard regression, xem §5).
- **`label_phase1.py`** đọc `integrated.parquet`, gọi `build_labels` + `validate`,
  ghi `labels.parquet` và `_label_log.json` (kèm `tie_counts`, `tail_nan`, `pct_pos`).
  Bắt exception → ghi log lỗi, exit 1 (tiện Phase 2).

---

## 5. Bốn check (hợp đồng chất lượng output)

| Check | Khẳng định |
| :--- | :--- |
| `spine_aligned` | `date` trùng khớp panel: tăng nghiêm ngặt, không trùng, đúng tập 1994 phiên |
| `label_domain` | Mỗi nhãn (bỏ NaN) chỉ nhận `∈ {-1, +1}`, không giá trị lạ |
| `tail_nan_exact` | Số NaN mỗi horizon **đúng bằng `k`** và **nằm trọn ở đuôi** (không hố giữa, không leading) |
| `tie_convention` | Mọi tie (`P_{t+k}=P_t`) được gán **+1**. **Guard**: khi spine = 1994 phiên, số đếm tie phải khớp EDA `{155,46,21,11}` |

**Vì sao `tie_convention` có 2 tầng.** Phần *invariant* (tie→+1) luôn assert. Phần
*đối chiếu số đếm 155/46/21/11* chỉ kích hoạt khi `n==1994` (snapshot khóa Phase 1).
Lý do: data còn fetch live (đến 30/06/2026) và Phase 2 chạy lại định kỳ — spine lớn lên
thì các con số này đổi, nếu hard-assert sẽ làm script exit 1 mỗi lần refresh. Snapshot
guard giữ được lợi ích cross-validate với EDA *bây giờ* mà không thành footgun *sau này*.
Lỗi off-by-one trong `shift` vẫn bị `tail_nan_exact` bắt độc lập (đã test).

Lần chạy 29/05/2026 (snapshot 1994 phiên, `2018-06-04 → 2026-05-28`): cả 4 pass.
`tie_counts={1:155, 5:46, 10:21, 20:11}` (khớp EDA), `tail_nan={1,5,10,20}`.
`pct_pos` (tỷ lệ +1, = rào baseline majority-class): **k=1: 54.74% · k=5: 54.50% ·
k=10: 54.08% · k=20: 55.98%**. Khác với Phase-0 window (54.4/52.8/51.1/51.0) vì đây là
toàn chuỗi 1994 phiên — TCB có drift tăng ròng trên cả giai đoạn nên `pct_pos` ở mọi
horizon đều ~54–56% (không suy về ~50% như cửa sổ đi-ngang Phase-0). Đây là **rào model
phải vượt**, không phải tín hiệu dự đoán.

---

## 6. Lưu ý & việc còn nợ

- **Output là interim**, chưa phải `features.parquet`: còn nguyên 1994 phiên (chưa cắt
  warmup) và chưa ráp với 20 feature. Cả hai là việc **Bước 7**.
- `pct_pos` là số *thông tin* (rào baseline), không phải tín hiệu dự đoán — báo cáo kèm
  tie-rate ở `k` nhỏ để không nhầm accuracy với predictability (`data.md` EDA §4).

**Tiếp theo — Bước 7 (Lắp ráp, xử lý NA & Xuất artifact):** merge `features_raw.parquet`
(Bước 5) + `labels.parquet` (Bước 6) theo `date`; cắt vùng warmup đầu chuỗi (~giữa 2019,
ràng buộc momentum 252 phiên); đảm bảo `data/processed/features.parquet` không còn NaN
trong vùng khả dụng (trừ nhãn đuôi) và `date` không trùng. Không PCA, không feature
selection (giữ tính confirmatory).