# Notebook 00 — Data inventory test plan

## Mục tiêu

Notebook `00_data_inventory.ipynb` trả lời hai câu hỏi:

> Một bảng có cấu trúc gì?

> Một dòng của bảng đại diện cho cái gì?

Notebook chỉ orchestration và ghi nhận xét. Mọi phép kiểm tra hoặc tổng hợp
được thực hiện bởi các public function có thể test độc lập bằng Polars
DataFrame/LazyFrame nhỏ.

## Trạng thái

```text
Artifact: acceptance-test specification
Notebook: 00_data_inventory.ipynb
Executable tests: tests/test_inventory.py
Product code: src/propensity/eda/inventory.py
```

## Flow của notebook

```text
scan_csv(path)
→ lấy DatasetSpec của bảng
→ spec.validate(lf)
→ build_table_inventory(lf, spec)
→ display structured results
→ ghi nhận xét cấu trúc trong Markdown
```

Inventory cần cung cấp:

- shape;
- physical dtype;
- semantic role;
- entity/time column nếu được khai báo;
- grain columns;
- key cardinality;
- preview nhỏ.

## Ngoài phạm vi

- Missing-value profiling, NaN và NA token.
- Exact duplicate rows của toàn bảng.
- Constant/all-missing columns.
- Business-rule validation, ví dụ `SALES_VALUE < 0`.
- Entity history, repeat entity và records per entity.
- Temporal coverage, missing `DAY` và records over time.
- Entity × time matrix.
- Relationship, foreign key, orphan và join coverage giữa các bảng.
- Target, propensity design và feature engineering.
- Plotting hoặc tạo Matplotlib figure.

Các nội dung trên thuộc notebook sau, không được đưa vào API của data
inventory chỉ vì tiện triển khai.

## Public API

```python
scan_csv(path, *, schema_overrides=None) -> pl.LazyFrame

DatasetSpec.validate(lf) -> None

table_overview(lf, spec) -> TableOverview
inspect_columns(lf, spec) -> pl.DataFrame
summarize_keys(lf, spec) -> pl.DataFrame
validate_grain(lf, grain_cols) -> GrainReport
preview(lf, n=5) -> pl.DataFrame
build_table_inventory(lf, spec) -> TableInventory
```

`build_table_inventory` chỉ orchestration các computation function phía trên.
Report object dùng `@dataclass(frozen=True, slots=True)`; các bảng chi tiết dùng
Polars DataFrame với schema cột cố định.

## Fixtures dùng chung

### `valid_transaction`

Một `pl.LazyFrame` nhỏ:

| household_key | BASKET_ID | DAY | PRODUCT_ID | QUANTITY | SALES_VALUE |
|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 1 | 10 | 1 | 5.0 |
| 1 | 100 | 1 | 11 | 2 | 8.0 |
| 2 | 101 | 3 | 10 | 1 | -10.0 |

Fixture thể hiện:

- `household_key` là entity column;
- `DAY` là time column dạng ordinal integer;
- một dòng là một product line trong basket;
- candidate grain `BASKET_ID + PRODUCT_ID` là unique;
- `BASKET_ID` lặp giữa nhiều product lines nên không phải row grain khi đứng
  một mình;
- giá trị âm là business anomaly, không phải structural failure.

### `duplicate_grain_transaction`

```text
(BASKET_ID=100, PRODUCT_ID=10)
(BASKET_ID=100, PRODUCT_ID=10)
(BASKET_ID=101, PRODUCT_ID=20)
```

### `missing_required_transaction`

Một `pl.LazyFrame` thiếu một required/grain column, ví dụ `BASKET_ID`.

### `dimension_table`

Một bảng nhỏ không khai báo entity/time column, dùng để bảo vệ behavior cho
dimension/lookup table.

## 1. Loader contract

### L1 — `scan_csv` trả LazyFrame

**Given** một CSV hợp lệ tồn tại.

**When** gọi `scan_csv(path)`.

**Then** function chỉ dựng lazy query plan bằng `pl.scan_csv` và trả
`pl.LazyFrame`; không dùng eager `pl.read_csv`, không gọi `collect`, không
print/display.

Test cần spy/mock đường gọi I/O; chỉ assert kiểu trả về là chưa đủ vì
`pl.read_csv(...).lazy()` vẫn trả `LazyFrame` sau khi đã đọc toàn bộ file.

### L2 — Schema override được truyền xuống Polars

**Given** một cột cần dtype override.

**When** gọi `scan_csv(path, schema_overrides=...)`.

**Then** override được chuyển cho `pl.scan_csv`; loader không tự cast dữ liệu
sau khi collect.

### L3 — File không tồn tại fail rõ ràng

**Given** path không trỏ đến file.

**When** gọi `scan_csv(path)`.

**Then** raise `FileNotFoundError` có path; không trả empty LazyFrame và không
tự tìm file gần giống.

## 2. DatasetSpec validation contract

Contract transaction hiện tại:

```text
name          = transaction
entity_col    = household_key
time_col      = DAY
time_grain    = day_index
grain_cols    = (BASKET_ID, PRODUCT_ID)
required_cols = QUANTITY, SALES_VALUE
```

### S1 — Spec mô tả đúng structural contract

**Given** chưa cần đọc dữ liệu thật.

**When** lấy `TRANSACTION_SPEC`.

**Then** các field name, entity, time, time grain, grain và required columns
đúng contract phía trên.

### S2 — Required columns bao gồm structural roles

**Given** `TRANSACTION_SPEC`.

**When** đọc `required_columns`.

**Then** kết quả chứa đúng một lần mỗi column sau:

```text
household_key
DAY
BASKET_ID
PRODUCT_ID
QUANTITY
SALES_VALUE
```

### S3 — Bảng đủ required columns pass validation

**Given** `valid_transaction`.

**When** gọi `TRANSACTION_SPEC.validate(lf)`.

**Then** không raise và input không bị cast, rename, thêm, xóa hoặc thay giá
trị.

### S4 — Thiếu required column fail rõ ràng

**Given** `missing_required_transaction`.

**When** gọi `TRANSACTION_SPEC.validate(lf)`.

**Then** raise `DatasetSpecError` có tên column thiếu.

### S5 — Entity/time là optional với bảng không có vai trò đó

**Given** spec của `dimension_table` có `entity_col=None` và `time_col=None`.

**When** validate bảng.

**Then** validation vẫn pass nếu các required columns khác đầy đủ; inventory
không tự đoán entity/time column.

## 3. Table overview contract

### O1 — Shape được tính đúng

**Given** `valid_transaction` có ba dòng và sáu cột.

**When** gọi `table_overview`.

**Then** report có `table_name="transaction"`, `n_rows=3` và `n_cols=6`.

### O2 — Overview chỉ chứa thông tin cấp bảng

**Given** một bảng hợp lệ.

**When** gọi `table_overview`.

**Then** report không chứa missing statistics, distribution, business-quality
flag hoặc plot.

### O3 — Input rỗng vẫn có shape hợp lệ

**Given** một LazyFrame có schema nhưng không có row.

**When** gọi `table_overview`.

**Then** `n_rows=0`, `n_cols` vẫn bằng số cột trong schema và không raise.

## 4. Column inventory contract

Schema output tối thiểu của `inspect_columns`:

```text
position: UInt32
column: String
physical_dtype: String
semantic_role: String
is_required: Boolean
is_grain: Boolean
```

### C1 — Mỗi input column có đúng một output row

**Given** `valid_transaction`.

**When** gọi `inspect_columns`.

**Then** số dòng output bằng số cột input; không thiếu và không lặp column.

### C2 — Physical dtype được giữ riêng với semantic role

**Given** `BASKET_ID` và `PRODUCT_ID` có physical dtype integer nhưng được spec
gán semantic role ID.

**When** gọi `inspect_columns`.

**Then** physical dtype vẫn là integer và semantic role là ID; hai cột không
bị mô tả như numerical measure.

### C3 — Required và grain flags theo spec

**Given** `TRANSACTION_SPEC`.

**When** gọi `inspect_columns`.

**Then** `BASKET_ID` và `PRODUCT_ID` có `is_required=True` và `is_grain=True`;
`SALES_VALUE` required nhưng không phải grain.

### C4 — Entity/time được phản ánh khi có khai báo

**Given** transaction spec khai báo `household_key` và `DAY`.

**When** gọi `inspect_columns`.

**Then** semantic role tương ứng là entity và time.

### C5 — Không tự suy đoán entity/time

**Given** `dimension_table` và spec không khai báo entity/time.

**When** gọi `inspect_columns`.

**Then** output không tự gán entity/time chỉ dựa vào tên hoặc dtype.

## 5. Key cardinality contract

Schema output tối thiểu của `summarize_keys`:

```text
column: String
semantic_role: String
n_rows: Int64
n_valid: Int64
n_null: Int64
n_unique: Int64
```

### K1 — Chỉ semantic keys được tổng hợp

**Given** transaction có entity, time và ID columns.

**When** gọi `summarize_keys`.

**Then** output chỉ tổng hợp entity/ID keys đã khai báo; time column `DAY` và
numerical measures như `QUANTITY`, `SALES_VALUE` không xuất hiện.

### K2 — Cardinality được tính đúng

**Given** `valid_transaction`.

**When** gọi `summarize_keys`.

**Then**:

```text
household_key: n_unique = 2
BASKET_ID:     n_unique = 2
PRODUCT_ID:    n_unique = 2
```

### K3 — Null key được báo, không bị loại âm thầm

**Given** một semantic key có một null.

**When** gọi `summarize_keys`.

**Then** `n_null=1`, `n_valid=n_rows-1`; input không bị drop hoặc fill.

## 6. Grain contract

### G1 — Grain columns đến từ spec

**Given** `TRANSACTION_SPEC`.

**When** build inventory.

**Then** candidate grain là `("BASKET_ID", "PRODUCT_ID")`; EDA logic không có
nhánh `if table == "transaction"` để hard-code grain.

### G2 — Unique grain được báo unique

**Given** `valid_transaction`.

**When** gọi `validate_grain(lf, spec.grain_cols)`.

**Then** report có:

```text
n_rows = 3
n_unique = 3
n_duplicate_rows = 0
is_unique = True
```

### G3 — Duplicate grain được báo, không raise

**Given** `duplicate_grain_transaction`.

**When** gọi `validate_grain`.

**Then** report có `n_rows=3`, `n_unique=2`, `n_duplicate_rows=1` và
`is_unique=False`; input vẫn giữ ba dòng.

### G4 — Empty grain configuration fail khi gọi trực tiếp

**Given** `grain_cols=()`.

**When** gọi trực tiếp `validate_grain`.

**Then** raise typed grain-configuration error. `build_table_inventory` được
phép trả `grain=None` nếu spec không khai báo grain.

### G5 — Grain column không tồn tại fail rõ ràng

**Given** grain tham chiếu một column không có trong input.

**When** gọi `validate_grain`.

**Then** raise typed analysis-column error có tên column thiếu.

## 7. Preview contract

### P1 — Preview mặc định có tối đa năm dòng

**Given** một LazyFrame có nhiều hơn năm dòng.

**When** gọi `preview(lf)`.

**Then** trả `pl.DataFrame` gồm năm dòng đầu và giữ nguyên thứ tự cột.

### P2 — Preview tôn trọng `n`

**Given** một LazyFrame có đủ dòng.

**When** gọi `preview(lf, n=2)`.

**Then** chỉ collect hai dòng đầu, không collect toàn bộ bảng rồi mới cắt.

### P3 — Preview không biến đổi dữ liệu

**Given** dữ liệu có giá trị âm hoặc null.

**When** gọi `preview`.

**Then** giá trị được giữ nguyên; function không clean, cast hoặc fill.

## 8. Combined inventory contract

### I1 — Inventory tập hợp đủ kết quả cần cho notebook

**Given** `valid_transaction` và `TRANSACTION_SPEC`.

**When** gọi `build_table_inventory`.

**Then** `TableInventory` chứa:

```text
overview
columns
keys
grain
preview
```

### I2 — Combined function chỉ orchestration

**Given** một LazyFrame và spec đã được truyền vào.

**When** gọi `build_table_inventory`.

**Then** function không đọc file, không print/display, không vẽ và không ghi
artifact.

### I3 — Kết quả đủ để diễn giải row meaning

**Given** transaction inventory có grain unique trên
`BASKET_ID + PRODUCT_ID`.

**When** notebook hiển thị inventory.

**Then** người viết notebook có đủ evidence để ghi nhận xét:

> Một dòng đại diện cho một product line trong một basket.

Function chỉ trả evidence có cấu trúc; câu nhận xét vẫn thuộc Markdown của
notebook, không được hard-code trong EDA module.

## 9. Vertical-slice integration

### V1 — Flow data inventory chạy xuyên suốt

**Given** một transaction CSV nhỏ và `TRANSACTION_SPEC`.

**When** chạy:

```text
scan_csv
spec.validate
build_table_inventory
```

**Then** mọi bước hoàn thành với structured output; loader vẫn lazy cho đến khi
các computation cần thiết được collect.

### V2 — Không side effect ngoài loader boundary

**Given** các public function inventory.

**When** chạy unit tests bằng LazyFrame nhỏ tự dựng.

**Then** không function nào đọc file, print, display, ghi artifact hoặc tạo
figure. Chỉ `scan_csv` thực hiện I/O.

## Ràng buộc cho bước implement sau

1. Một function làm một việc và không có side effect.
2. Loader/I/O tách khỏi inventory computation.
3. Report dùng `@dataclass(frozen=True, slots=True)`; DataFrame chi tiết có
   schema cột cố định.
4. Dataset-specific facts nằm trong `DatasetSpec`, không hard-code trong hàm.
5. Mỗi public function có happy case và ít nhất một boundary/adverse case.
6. Không thêm quality, temporal, entity-history hoặc relationship logic vào
   inventory.
7. Không tạo module rỗng hoặc abstraction chưa có caller thực tế.
8. Public API thay đổi theo hướng additive; rename giữ compatibility facade
   trong giai đoạn migrate.
9. Trước khi hoàn tất implementation phải chạy `pytest`, `ruff check` và
   `ruff format --check`.

## Quyết định đã chốt

- `summarize_keys` chỉ gồm entity/ID columns; time column không phải key.
- Public API dùng tên `validate_grain`.
- `inspect_columns` dùng schema:
  `position`, `column`, `physical_dtype`, `semantic_role`, `is_required`,
  `is_grain`.
- `summarize_keys` dùng schema:
  `column`, `semantic_role`, `n_rows`, `n_valid`, `n_null`, `n_unique`.
- `null_pct`, `unique_pct` và `is_unique` không nằm trong key summary.
- `TableOverview` lưu `table_name` cùng `n_rows` và `n_cols`.

