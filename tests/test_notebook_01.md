# Test specification — Notebook 01: Feature Understanding

## Mục tiêu

Notebook 01 phải tạo bằng chứng để trả lời cho từng cột của
`transaction_data.csv`:

1. physical dtype, `ColumnRole` và analysis route;
2. `n_null` và `n_unique`;
3. phân phối numeric: min–max, skewness, quantile, IQR và hai đuôi;
5. phạm vi và độ phủ temporal.

Business meaning, unit và ý nghĩa null được đối chiếu với data dictionary trong
phần kết luận notebook. Chúng không phải metadata bắt buộc của package và không
được dùng để route phép tính.

## Nguyên tắc cấu hình

- Physical dtype được điều chỉnh ở `TRANSACTION_SCHEMA_OVERRIDES`.
- Structural role được suy ra bởi `DatasetSpec` và `ColumnRole`.
- Analysis route mặc định được suy ra từ `ColumnRole`.
- Chỉ encoding không thể suy ra mới xuất hiện trong
  `TRANSACTION_ANALYSIS_OVERRIDES`:
  - `TRANS_TIME -> clock_time`;
  - `WEEK_NO -> ordinal_time`.
- Không khai báo definition cho đủ 12 cột.
- Chưa kiểm tra override thừa, thiếu hoặc trùng trong notebook này.

## Contract của feature catalog

```text
position: UInt32
column: String
physical_dtype: String
structural_role: String
analysis_kind: String
```

### Routing mặc định

| `ColumnRole` | `AnalysisKind` |
|---|---|
| `ENTITY`, `ID` | `identifier` |
| `NUMERIC`, `TARGET` | `numeric` |
| `CATEGORICAL`, `BOOLEAN` | `categorical` |
| `DATETIME` | `datetime` |
| `TIME` | `ordinal_time` |
| `OTHER` | yêu cầu override |

## Bộ test tự động

### Catalog và routing

**Given** LazyFrame transaction có đủ 12 cột và `TRANSACTION_SPEC`.

**When** gọi `build_feature_catalog()` với hai analysis overrides.

**Then**:

- catalog có đúng schema cố định và đủ 12 dòng;
- ID, numeric và `DAY` được route tự động;
- `TRANS_TIME`, `WEEK_NO` dùng route override;
- cấu hình transaction chỉ chứa đúng hai ngoại lệ.

### Cardinality

Kiểm tra:

- output chỉ gồm `column`, `n_null`, `n_unique`;
- cột all-null có `n_null = n_rows` và `n_unique = 0`.

### Data quality

Sentinel và giá trị bất khả thi không nằm trong feature-understanding report.
Kiểm tra riêng `TRANS_TIME=2415` được ghi vào data quality nhưng dữ liệu gốc
không bị thay đổi.

### Numeric distribution

Chỉ năm measure numeric xuất hiện:

- `QUANTITY`;
- `SALES_VALUE`;
- `RETAIL_DISC`;
- `COUPON_DISC`;
- `COUPON_MATCH_DISC`.

Kiểm tra min–max, skewness, quantile, IQR và số giá trị ngoài hai fence.

### Temporal coverage

Chỉ `DAY`, `TRANS_TIME`, `WEEK_NO` xuất hiện. Với ordinal time, kiểm tra min,
max, span, số period quan sát và coverage. Với clock time, không tính coverage
như một dãy số liên tục.

### Combined report

`build_feature_understanding()` phải trả đủ bốn bảng có contract ổn định:

- `catalog`;
- `cardinality`;
- `numeric_profiles`;
- `temporal_profiles`.

## Ngoài phạm vi

- tự sửa giá trị bất khả thi;
- scale hoặc encode feature;
- áp quy tắc loại outlier;
- kiểm tra metadata/override thừa, thiếu, trùng;
- categorical frequency table (sẽ bổ sung khi làm nhóm categorical).

## Ranh giới với behavioral EDA

Notebook 01 dừng ở feature-level profiling. Technical EDA sau khi đổi grain
được tách sang:

- `03_transaction_behavior.ipynb`: identifier concentration, basket,
  household transaction metrics, store/discount và correlation chọn lọc;
- `04_temporal_household_behavior.ipynb`: activity theo `DAY`/`WEEK_NO`,
  active periods, household span và purchase intervals.

Khoảng quay lại phải dùng distinct `(household_key, DAY)`. Nhiều basket trong
cùng ngày không được tạo khoảng cách bằng 0.
