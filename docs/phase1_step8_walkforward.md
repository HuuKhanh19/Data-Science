# Step 8 — (Cầu nối) Walk-Forward split & Z-score per-window

> Đây là **cầu nối**, không phải một Step xử lý dữ liệu tĩnh như Step 1→7. Step 1→7
> mỗi cái nuốt file vào, nhả một artifact bền ra. Step 8 **không ghi file** — nó là
> module logic thuần `src/model/walk_forward.py` chạy *bên trong* vòng lặp refit của
> Step 11, biến `features.parquet` thành các cặp `(train, test)` per-window phù du trong
> RAM. Nó đóng 2 chốt chặn leakage cuối cùng: **buffer gap `k`** (chốt 3) và **Z-score
> fit-trên-train** (chốt 4).

---

## 1. Vì sao không ghi file tĩnh

Hai lý do bản chất:

1. **Không có "tập đã chia" duy nhất.** Walk-forward refit hàng tuần → sinh **149–153 cặp
   train/test** (tùy `k`), mỗi tuần một cặp. Không tồn tại một file "đã split" để lưu.
2. **Z-score phải fit per-window.** `μ,σ` tính riêng trên train window từng tuần. Nếu
   chuẩn hóa sẵn thành một file tĩnh, ta buộc dùng `μ,σ` toàn cục (gồm cả tương lai) →
   rò rỉ thống kê. Vì thế `features.parquet` cố tình giữ feature ở dạng **thô chưa chuẩn
   hóa**; việc scale dời hẳn về đây.

Hệ quả: Step 8 kiểm chứng bằng **bất biến logic** (§6) chứ không bằng `_*_log.json` kiểu
Step 1→7.

---

## 2. Hình học một bước walk-forward

Mọi mốc tính theo **chỉ số phiên** (vị trí) trên vùng khả dụng `[0, 1742)`, không theo
ngày lịch. Test block = một **tuần lịch ISO**. Với tuần có phiên đầu ở vị trí `T_w`:

```
train = [T_w − 1000 − k, T_w − k)   ← đúng 1000 phiên
gap   = [T_w − k,        T_w)        ← k phiên bị LOẠI
test  = các phiên của tuần đó        ← bắt đầu ở T_w
```

**Vì sao có gap `k`.** Nhãn `y_{t,k} = sign(P_{t+k} − P_t)` cần biết giá ở phiên `t+k`.
Tại thời điểm refit (đầu tuần `w`, mốc `T_w`), phiên gần nhất *thật sự* đã quan sát đủ
nhãn là `t = T_w − k` (vì `y_{T_w−k,k}` chỉ cần `P_{T_w}`, đã có). Mọi phiên
`t ∈ (T_w−k, T_w)` có nhãn phụ thuộc giá *sau hôm nay* → chưa biết → để vào train là nhìn
trộm tương lai. Cắt đúng `k` phiên đó là **chốt leakage 3**. Đây là vấn đề *per-window*,
khác với đuôi NaN *toàn cục* ở Step 6.

**Hệ quả mép đầu.** Train luôn giữ đúng 1000 phiên (rolling, không expanding), nên tuần
test đầu tiên chỉ phát được khi `T_w ≥ 1000 + k`. Vì vậy `k` phiên đầu của vùng test danh
nghĩa `[1000, 1742)` không được dự đoán, và số tuần refit giảm nhẹ khi `k` lớn.

---

## 3. Mốc Phase-0/test đã chốt (số thật, 29/05/2026)

Chạy `python -m src.model.walk_forward data/processed/features.parquet` để đối chiếu một
lần — đây là việc giải quyết cái ⚠ tồn từ `research_design.md §3.1`:

| Hạng mục | Giá trị |
| :--- | :--- |
| Vùng khả dụng | 1742 phiên, 2019-06-07 → 2026-05-28 |
| Phase-0 phiên cuối (index 999) | **2023-06-05** |
| Test phiên đầu / mốc (index 1000) | **2023-06-06** |
| Tuần refit `k`=1/5/10/20 | **153 / 152 / 151 / 149** |
| Tuần đánh giá đầu theo `k` | 2023-06-12 / 06-19 / 06-26 / 07-10 |

Ranh giới định nghĩa bằng **chỉ số phiên** (Phase-0 = `[0,1000)`), bất biến sau khi chốt;
ngày chỉ là annotation.

---

## 4. Giải thích mã nguồn

```
src/model/walk_forward.py        ← module logic thuần, không I/O bền
├── N_TRAIN = 1000, LABEL_COLS    ← hằng số khóa
├── feature_columns(df)           ← 20 cột feature = mọi cột trừ `date` + 4 nhãn
├── iso_week_blocks(dates)        ← gom phiên liên tiếp theo tuần lịch ISO
├── walk_forward_splits(...)      ← generator (train_idx, test_idx) — THUẦN index
├── drop_label_nan(X, y)          ← loại dòng nhãn NaN
├── scale_train_test(X_tr, X_te)  ← Z-score fit-trên-train
├── build_window(...)             ← ráp 1 window: chọn cột → lọc NaN → z-score
├── phase0_boundary_date(dates)   ← ngày mốc index 1000 (đối chiếu §3.1)
└── __main__                      ← in mốc + số tuần refit (chạy đối chiếu một lần)
```

Tách bạch trách nhiệm: **splitter chỉ tính hình học chỉ số, không đụng tới nhãn**; lọc NaN
và chuẩn hóa nằm ở `build_window`. Nhờ vậy splitter tái dùng được cho cả Step 9 (baselines),
Step 11 (train+infer) và inference live ở Step 13.

### `iso_week_blocks(dates)`
Đổi `dates` thành `DatetimeIndex`, lấy `(iso_year, iso_week)` của từng phiên qua
`.isocalendar()`, rồi quét tuyến tính gom các phiên **liên tiếp cùng khóa tuần** thành một
block chỉ số. Vì `date` tăng nghiêm ngặt (đảm bảo bởi `date_key` check ở Step 7), các phiên
cùng tuần luôn nằm kề nhau → một lượt quét là đủ. Tuần có ngày nghỉ HOSE → block ngắn hơn 5.
Khóa `(iso_year, iso_week)` xử lý đúng ranh giới năm (tuần 52/53 vắt qua giao thừa).

### `walk_forward_splits(dates, k, n_train=1000)`
Generator. Với mỗi block tuần (lấy `t0 = test_idx[0]`): tính `train_start = t0 − n_train − k`;
**bỏ qua** tuần nếu `train_start < 0` (train chưa đủ chỗ → đây là cơ chế "tuần đầu chỉ phát
khi `T_w ≥ 1000+k`"); ngược lại `yield (arange(train_start, t0−k), test_idx)`. Khoảng
`[train_start, t0−k)` đúng `n_train` phiên; khoảng `[t0−k, t0)` là `k` phiên gap bị nhảy qua.
Hàm chỉ nhận `dates`, **không** nhận DataFrame hay nhãn.

### `scale_train_test(X_train, X_test)` — chốt leakage 4
Fit `StandardScaler` (`μ,σ`) **chỉ trên `X_train`**, rồi `transform` cả train lẫn test bằng
đúng bộ `μ,σ` đó. Trả thêm `scaler` để Step 11/12 log lại tham số nếu cần. Không bao giờ
fit trên test.

### `build_window(df, train_idx, test_idx, label_col, feature_cols=None)`
Ráp một window hoàn chỉnh, là interface mà Step 11 gọi mỗi tuần:
chọn cột feature (mặc định = 20 cột thô) và `label_col` (vd. `"y_5"`) → **lọc NaN nhãn ở
train** bằng `drop_label_nan` → **z-score** → trả
`(X_train_z, y_train, X_test_z, y_test, test_dates, scaler)`. Buffer gap đã đảm bảo nhãn
train luôn quan sát được nên bước lọc NaN ở train chỉ là lưới an toàn; `y_test` có thể còn
NaN ở đuôi chuỗi (giữ cho inference, loại khi đánh giá).

### `phase0_boundary_date(dates, n_phase0=1000)`
Trả ngày tại vị trí index `n_phase0` = phiên đầu tiên **sau** Phase-0 `[0,1000)`. Dùng để
in mốc đối chiếu (§3).

---

## 5. Quan hệ với leakage

Step 8 đóng nốt **2/4 chốt chặn** còn lại (2 chốt kia đã đóng ở Step 4–5):

| Chốt | Cơ chế | Hàm |
| :--- | :--- | :--- |
| (3) Buffer gap `k` | Loại `k` phiên cuối train (nhãn chưa quan sát) | `walk_forward_splits` |
| (4) Z-score fit-trên-train | `μ,σ` chỉ từ train window mỗi tuần | `scale_train_test` |

Sau Step 8, cả 4 chốt đã đóng: (1) lag `≤t-1` · (2) as-of `release_date` · (3) buffer gap ·
(4) z-score per-window.

---

## 6. Bất biến module đảm bảo (hợp đồng)

Trong lúc phát triển, các bất biến dưới đây được kiểm bằng unit test trên chuỗi giả lập
(đã gỡ bỏ để giữ repo gọn); chúng là hợp đồng mà mọi thay đổi sau này phải giữ:

| Bất biến | Khẳng định |
| :--- | :--- |
| Train length | `len(train_idx) == 1000` ở mọi window |
| Buffer gap đúng | `test_idx[0] − train_idx[-1] == k + 1` (đúng `k` phiên bị bỏ) |
| No leakage | `max(train_idx) < min(test_idx)` |
| Tuần đầu hợp lệ | Tuần test đầu có `T_w ≥ 1000 + k`; tuần trước đó bị loại |
| Test block = 1 tuần ISO | Mọi chỉ số trong một `test_idx` cùng `(iso_year, iso_week)` |
| Block rời & tăng | Các `test_idx` không chồng lấn, mép tăng dần |
| Z-score fit-trên-train | `scaler.mean_ == X_train.mean()`, KHÁC mean toàn chuỗi; train sau scale ~ N(0,1) |
| Train sạch NaN | `build_window` trả `y_train` không còn NaN |
| Phòng thủ input | `dates` không tăng nghiêm ngặt / có trùng → `ValueError` |