# Session logs

Tài liệu append-only ghi lại process của mỗi session implementation. Không sửa retroactively.

**Khác biệt với các tài liệu khác**:

| Tài liệu | Lifecycle | Mục đích |
|---|---|---|
| `research_design.md` | Lock end of Tuần 2 (immutable trừ bug fix) | WHAT was analyzed — methodology pre-registered |
| `IMPLEMENTATION.md` | Working doc, sửa khi architectural contract thay đổi | Current state of code architecture |
| `docs/sessionXX_*.md` | **Append-only**, mỗi session 1 file | Process record — what happened, when, why, with detail |
| `CHANGELOG.md` | Append per version bump | High-level version history |

## Index

| Session | File | Phase | Status |
|---|---|---|---|
| 01 | [session01_data_acquisition.md](session01_data_acquisition.md) | Phase 1 + Phase 0 partial | ✅ Complete |
| 02 | [session02_asof_join.md](session02_asof_join.md) | Phase 1 | ✅ Complete |
| 03 | (planned) `session03_loaders_and_macro_acquisition.md` | Phase 1 | 🔄 Next |

## Template structure (7 sections)

Mỗi session doc theo cấu trúc cố định, mức chi tiết khác nhau tùy session type:

### 1. Mục đích và bối cảnh
- Vị trí trong pipeline tổng thể
- Pre-conditions
- Câu hỏi cần trả lời (open questions từ IMPLEMENTATION)
- Deliverables expected

### 2. Source code — mô tả chi tiết
Cho từng file/module được tạo hoặc sửa đổi quan trọng:
- **Triết lý kiến trúc** — design philosophy
- **Public API** — function signatures, purpose, parameters
- **Internal helpers** — supporting functions
- **Quyết định quan trọng** — design choices với rationale (citation hoặc principle)
- Code excerpts cho phần illustrative

### 3. Data (chỉ áp dụng cho session xử lý dữ liệu)
- **Input data sources** — API, schema, units, conventions
- **Processing pipeline** — sequence các step (diagram ASCII OK)
- **Quality checks** — mapping về Slide_Data_Science.md 5 mức data quality
- **Output data** — schemas thực tế, summary stats, sample rows nếu cần
- **Quyết định về data** (canonical unit, encoding, normalization conventions)

### 4. Model (chỉ áp dụng cho session implement model)
- **Architecture** — diagram, components, parameter count
- **Training algorithm** — loss, optimizer, schedule, early stopping
- **Hyperparameters** — values, source (tuned vs fixed), citations
- **Inference flow** — input pre-processing → forward → post-processing
- **Pre-conditions** — required data shape, normalization
- **Output** — prediction format, confidence/probability handling

### 5. Verification và experiments
- **Methodology** từng experiment
- **Results** — tables, distributions, observed values
- **Interpretation** — kết luận khoa học (không speculation)
- **Locked thresholds** từ experiments

### 6. Issues encountered và fixes
Bảng các bug, theo thứ tự thời gian:
- Symptom (error message hoặc behavior observed)
- Root cause (technical explanation)
- Fix applied
- Lesson learned (generalizable insight)

### 7. Open questions / next session
- **Resolved** trong session này (mapping về IMPLEMENTATION §11)
- **Locked baselines** cho Phase 0+
- **Mở ra** cho session sau
- **Next session preview** — module + critical points

## Nguyên tắc viết

- **Pedagogical**: đọc được standalone, không cần đọc chat history
- **Có cơ sở**: mọi design decision tham chiếu Slide_Data_Science.md, research_design.md, hoặc literature
- **Đầy đủ stats**: không chỉ "data acquired" mà cần exact n_rows, ranges, distributions
- **Honest về limitations**: nếu chưa verify một aspect, ghi rõ trong "mở ra cho session sau"
- **Reproducible**: ai đó đọc doc + code phải replicate được results
