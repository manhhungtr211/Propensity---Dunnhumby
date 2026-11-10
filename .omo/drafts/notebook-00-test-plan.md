# Notebook 00 — Transaction table-understanding test plan

## Trạng thái

```text
Artifact: acceptance-test draft
Table: transaction_data
Executable tests: chưa viết
Product code: chưa sửa
Notebook code: ngoài phạm vi hiện tại
```

Tài liệu này chốt behavior của các public function mà notebook `00` sẽ dùng.
Sau khi được duyệt, mỗi acceptance criterion mới được chuyển thành test `pytest`
trước khi implementation được viết hoặc sửa.

## Trách nhiệm của notebook 00

Notebook chỉ orchestration và interpretation:

```text
scan transaction
→ apply TRANSACTION_SPEC
→ build table inventory
→ describe quality
→ describe entities
→ describe ordinal temporal coverage
→ ghi nhận xét trong Markdown
```

Notebook không chứa implementation của các phép đếm, grain check, missing check
hoặc temporal matrix.

## Ngoài phạm vi

- Không relationship/FK/orphan/join coverage.
- Không target hoặc propensity design.
- Không feature engineering.
- Không tạo calendar date giả từ `DAY`.
- Không kiểm tra business validity của giá trị như `SALES_VALUE = -10`.
- Không thiết kế lại plotting trong bước đầu.
- Không tạo `profiling.py` cho đến khi có function cụ thể cần tách.

## Fixtures dùng chung

### `valid_transaction`

Một `pl.LazyFrame` nhỏ, tự dựng, có ít nhất:

| household_key | BASKET_ID | DAY | PRODUCT_ID | QUANTITY | SALES_VALUE | COUPON_DISC |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 1 | 10 | 1 | 5.0 | 0.0 |
| 1 | 101 | 3 | 11 | 2 | 8.0 | -1.5 |
| 2 | 102 | 3 | 10 | 1 | -10.0 | 0.0 |

Fixture này cố ý bảo vệ các behavior:

- `DAY` là integer ordinal index.
- Day index `2` không xuất hiện.
- Household `1` lặp lại.
- Discount có decimal âm nên không được infer thành integer.
- `SALES_VALUE = -10` là business anomaly nhưng không phải structural failure.

### `duplicate_grain_transaction`

Ba dòng, trong đó một candidate grain lặp lại:

```text
(BASKET_ID=100, PRODUCT_ID=10)
(BASKET_ID=100, PRODUCT_ID=10)
(BASKET_ID=101, PRODUCT_ID=20)
```

### `missing_required_transaction`

Một LazyFrame thiếu một cột bắt buộc, ví dụ `BASKET_ID`.

## 1. Loader contract

Public function dự kiến:

```python
scan_dunnhumby_table(name: str) -> pl.LazyFrame
```

### L1 — Transaction được scan lazily

**Given** file `transaction_data.csv` tồn tại trong Dunnhumby raw directory.

**When** gọi `scan_dunnhumby_table("transaction")`.

**Then** kết quả là `pl.LazyFrame`; function không collect toàn bộ dữ liệu và
không print.

### L2 — Decimal discount được parse ổn định

**Given** transaction chứa `COUPON_DISC = -1.1` hoặc một decimal tương đương.

**When** collect projection nhỏ có `COUPON_DISC`.

**Then** collection thành công và physical dtype của `COUPON_DISC` hỗ trợ số
thập phân.

### L3 — Unknown table fail rõ ràng

**Given** table name không được loader hỗ trợ.

**When** gọi loader.

**Then** raise typed loader error chứa table name; không tự tìm một file gần
giống và không trả pandas DataFrame.

## 2. DatasetSpec grain contract

Field dự kiến:

```python
grain_cols: tuple[str, ...] = ()
```

### S1 — Grain mặc định là optional

**Given** `DatasetSpec` không truyền `grain_cols`.

**When** khởi tạo spec.

**Then** `grain_cols == ()`; các behavior schema hiện tại không đổi.

### S2 — Grain columns thuộc required raw columns

**Given** một spec có `grain_cols=("BASKET_ID", "PRODUCT_ID")`.

**When** đọc `required_columns`.

**Then** cả hai grain columns xuất hiện đúng một lần trong required columns.

### S3 — Thiếu grain column fail ở structural boundary

**Given** spec khai báo một grain column nhưng LazyFrame không có column đó.

**When** gọi `spec.validate(lf)`.

**Then** raise `DatasetSpecError` chỉ rõ column thiếu; validation không sửa dữ
liệu.

## 3. Inventory contract

Public functions dự kiến:

```python
table_overview(lf, spec) -> TableOverview
inspect_columns(lf, spec) -> pl.DataFrame
summarize_keys(lf, spec) -> pl.DataFrame
validate_grain(lf, grain_cols) -> GrainReport
build_table_inventory(lf, spec) -> TableInventory
```

Các object report là `@dataclass(frozen=True, slots=True)`; các bảng chi tiết
là Polars DataFrame với schema cột cố định.

### I1 — Table overview trả đúng cấu trúc cơ bản

**Given** `valid_transaction` và `TRANSACTION_SPEC`.

**When** gọi `table_overview`.

**Then** report trả đúng table name, row count và column count; không chứa
missing statistics, distribution hoặc plot.

### I2 — Column inventory phân biệt physical dtype và semantic role

**Given** `BASKET_ID` và `PRODUCT_ID` có physical dtype numeric nhưng được
override thành ID.

**When** gọi `inspect_columns`.

**Then** output có schema cố định tối thiểu:

```text
column
physical_dtype
semantic_role
is_required
is_grain
```

và hai ID không bị báo là numerical measure.

### I3 — Key summary báo cardinality theo semantic spec

**Given** transaction có entity key và các ID override.

**When** gọi `summarize_keys`.

**Then** chỉ các semantic key được tổng hợp; output có key name, role, valid
row count, null count và unique count.

### I4 — Unique grain được báo unique

**Given** mỗi combination trong `grain_cols` xuất hiện đúng một lần.

**When** gọi `validate_grain`.

**Then** trả `GrainReport` với:

```text
n_rows == n_unique
n_duplicate_rows == 0
is_unique is True
```

Function không raise và không print.

### I5 — Duplicate grain được báo, không bị sửa

**Given** `duplicate_grain_transaction`.

**When** gọi `validate_grain`.

**Then** report có `n_rows=3`, `n_unique=2`, `n_duplicate_rows=1`,
`is_unique=False`; input vẫn giữ nguyên ba dòng.

### I6 — Grain configuration rỗng fail rõ ràng khi gọi trực tiếp

**Given** `grain_cols=()`.

**When** gọi trực tiếp `validate_grain`.

**Then** raise typed grain-configuration error. `build_table_inventory` có thể
bỏ qua grain report khi spec không khai báo grain.

### I7 — Thiếu grain column fail rõ ràng

**Given** `grain_cols` tham chiếu một column không tồn tại.

**When** gọi `validate_grain`.

**Then** raise typed analysis-column error chứa tên column thiếu.

### I8 — High-level inventory chỉ orchestration

**Given** transaction hợp lệ.

**When** gọi `build_table_inventory`.

**Then** kết quả chứa structured table overview, column inventory, key summary
và optional grain report; function không đọc file, không vẽ và không print.

## 4. Quality contract

Public function notebook dùng:

```python
describe_quality(lf, spec) -> QualityReport
```

Missing/duplicate/constant helpers là private cho đến khi có caller thứ hai.

### Q1 — Missing summary nhận diện các dạng thiếu đã cấu hình

**Given** dữ liệu chứa null, NaN và string NA token.

**When** gọi `describe_quality`.

**Then** report đếm đúng missing theo column và toàn bảng.

### Q2 — Exact duplicate rows được đếm đúng

**Given** hai row giống hoàn toàn.

**When** gọi `describe_quality`.

**Then** report có đúng duplicate-row count và percentage.

### Q3 — Constant và all-missing columns được phân biệt

**Given** một constant non-null column và một all-null column.

**When** gọi `describe_quality`.

**Then** hai column được đưa vào đúng danh sách tương ứng.

### Q4 — Business anomaly không phải quality structural failure mặc định

**Given** `SALES_VALUE = -10` nhưng cấu trúc bảng hợp lệ.

**When** gọi `describe_quality`.

**Then** function hoàn thành; không raise, cast hoặc thay giá trị.

### Q5 — Quality report có cấu trúc ổn định

**Given** một LazyFrame hợp lệ.

**When** gọi `describe_quality`.

**Then** trả frozen structured report và `column_quality` DataFrame có schema
cột cố định; không trả raw dict tùy ý.

## 5. Entity contract

Public function:

```python
describe_entities(lf, spec) -> EntityReport
```

### E1 — Repeat entities được đếm đúng

**Given** household `1` có hai records và household `2` có một record.

**When** gọi `describe_entities`.

**Then** report có hai unique entities và một repeat entity.

### E2 — Missing entity records được báo riêng

**Given** một record có null `household_key`.

**When** gọi `describe_entities`.

**Then** record đó được tính vào missing-entity count và không tạo một valid
entity.

### E3 — History span dùng ordinal DAY trực tiếp

**Given** một household xuất hiện tại `DAY=1` và `DAY=3`.

**When** gọi `describe_entities`.

**Then** history span bằng `2` day-index units; không xuất hiện calendar date,
month hoặc base-date conversion trong output.

### E4 — Bảng không đủ entity/time configuration fail rõ ràng

**Given** spec thiếu column được khai báo làm entity hoặc time.

**When** gọi `describe_entities`.

**Then** raise typed analysis error; không fallback sang column khác.

## 6. Ordinal temporal contract

Public functions dự kiến:

```python
describe_temporal(lf, spec) -> OrdinalTemporalReport | CalendarTemporalReport
build_entity_time_matrix(lf, spec) -> EntityTimeMatrix
```

Notebook transaction nhận `OrdinalTemporalReport`; calendar report được giữ để
không phá các dataset Date/Datetime hiện có.

### T1 — Integer DAY được nhận diện là ordinal axis

**Given** `DAY` có dtype integer.

**When** gọi `describe_temporal`.

**Then** trả `OrdinalTemporalReport` với `axis_kind="ordinal"`; không gọi date
parser và không sinh ngày calendar.

### T2 — Ordinal bounds và gaps được tính đúng

**Given** observed DAY values `[1, 3, 3]`.

**When** gọi `describe_temporal`.

**Then**:

```text
start_index = 1
end_index = 3
observed_indices = 2
missing_indices = [2]
records_per_index = DAY 1→1 row, DAY 2→0 rows, DAY 3→2 rows
```

### T3 — Không suy ra month từ DAY

**Given** ordinal DAY values và `time_grain="1d"`.

**When** tạo temporal report.

**Then** output không chứa `start_date`, `end_date`, `history_months` hoặc
`missing_months`; `WEEK_NO` không được dùng để âm thầm map calendar.

### T4 — Missing DAY được báo nhưng raw value không bị sửa

**Given** một record có null DAY.

**When** gọi `describe_temporal`.

**Then** report tăng missing-time count; input không bị drop/cast/mutate.

### T5 — Calendar input vẫn dùng calendar report

**Given** một bảng khác có Date/Datetime time column.

**When** gọi cùng public function.

**Then** trả `CalendarTemporalReport`; đây là regression guard cho behavior cũ.

### T6 — Entity × DAY matrix được tính không cần plot

**Given** hai households hoạt động trên DAY 1 và DAY 3.

**When** gọi `build_entity_time_matrix`.

**Then** trả structured matrix/coverage result với day indices `1, 2, 3`, fill
zero cho entity-day không hoạt động, và không tạo Matplotlib `Figure`.

## 7. Transaction vertical-slice integration

### V1 — Flow table-understanding chạy xuyên suốt

**Given** transaction được scan qua Dunnhumby loader và `TRANSACTION_SPEC` đã
được duyệt.

**When** chạy lần lượt:

```text
spec.validate
build_table_inventory
describe_quality
describe_entities
describe_temporal
build_entity_time_matrix trên bounded sample/window
```

**Then** mọi bước hoàn thành với structured output; `DAY` vẫn integer và
`COUPON_DISC` collect được decimal.

### V2 — Không function phân tích nào làm I/O hoặc visualization

**Given** các function public trong inventory/quality/entity/temporal.

**When** chạy unit tests bằng LazyFrame ba dòng tự dựng.

**Then** không function nào đọc file, print, display, ghi artifact hoặc tạo
figure. Chỉ loader làm I/O; chỉ visualization tạo figure.

## Ràng buộc implementation sau khi AC được duyệt

1. Một function làm một việc và không có side effect.
2. Loader/I/O tách khỏi computation.
3. Report dùng `@dataclass(frozen=True, slots=True)`; bảng chi tiết dùng Polars
   DataFrame với schema cố định.
4. Không có `if table == "transaction"` trong EDA; khác biệt nằm trong spec và
   loader parse configuration.
5. Mỗi public function có happy case và ít nhất một adverse/boundary case.
6. YAGNI: không relationships, target, profiling file rỗng hoặc abstraction chỉ
   có một caller.
7. Chạy `pytest`, `ruff check`, `ruff format --check` và no-excuse audit trước
   khi hoàn tất.
8. Thay đổi public API theo hướng additive; rename phải giữ compatibility
   facade cho đến khi caller được migrate.
9. Chuyển package từ `src` sang `src/<package>/` là task riêng sau notebook 00,
   không trộn vào implementation này.

## Điểm cần chủ repo duyệt trước executable tests

- Candidate `TRANSACTION_SPEC.grain_cols` cụ thể.
- Danh sách decimal schema overrides chính xác của transaction loader.
- Tên public API cuối cùng (`validate_grain` hay `summarize_grain`).
- Schema cột cuối cùng của `inspect_columns`, key summary và quality detail.
- Việc `build_entity_time_matrix` thuộc temporal computation nhưng plotting API
  cũ chưa được migrate trong task này.
