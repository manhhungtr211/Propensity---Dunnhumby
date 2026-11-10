# Transaction schema test plan

## Mục tiêu

Tài liệu này mô tả các behavior cần được thống nhất trước khi viết test và code
cho schema của bảng `transaction`.

Schema chỉ chịu trách nhiệm về contract cấu trúc và semantic metadata. Schema
không đánh giá data quality hoặc tính hợp lệ của business value.

## Contract hiện tại

| Thuộc tính | Giá trị |
|---|---|
| Dataset name | `transaction` |
| Entity column | `household_key` |
| Time column | `DAY` |
| Time grain | `1d` |
| Required columns | `BASKET_ID`, `PRODUCT_ID`, `QUANTITY`, `SALES_VALUE` |
| ID overrides | `BASKET_ID`, `PRODUCT_ID` |
| Target column | Chưa khai báo |

Các cột bắt buộc đầy đủ là:

```text
household_key
DAY
BASKET_ID
PRODUCT_ID
QUANTITY
SALES_VALUE
```

## Acceptance criteria

### AC1 — DatasetSpec mô tả được contract của transaction

**Given** chưa cần đọc dữ liệu.

**When** khởi tạo transaction spec.

**Then** spec phải mô tả đúng:

- dataset name;
- entity column;
- time column;
- time grain;
- required columns;
- các semantic role override.

### AC2 — Dữ liệu đúng contract phải pass validation

**Given** bảng transaction có đầy đủ các cột bắt buộc.

**When** validate bảng bằng transaction spec.

**Then** validation hoàn thành mà không raise error và dữ liệu vẫn có thể được
sử dụng tiếp.

### AC3 — Thiếu required column phải fail rõ ràng

**Given** bảng transaction thiếu một cột bắt buộc, ví dụ `BASKET_ID`.

**When** validate bảng bằng transaction spec.

**Then** validation phải raise `DatasetSpecError` và thông báo phải chỉ ra
`BASKET_ID` đang thiếu.

Các cột cần cân nhắc kiểm tra riêng:

| Nhóm | Ví dụ |
|---|---|
| Entity column | `household_key` |
| Time column | `DAY` |
| Additional required column | `BASKET_ID` |

### AC4 — Business value bất thường không phải schema failure

**Given** bảng có đầy đủ cột bắt buộc nhưng chứa business value bất thường, ví
dụ `SALES_VALUE = -10`.

**When** validate bảng bằng transaction spec.

**Then** schema vẫn phải pass vì kiểm tra giá trị bán hàng thuộc data quality
hoặc business rule, không thuộc structural schema.

### AC5 — Validation không được sửa dữ liệu

**Given** một bảng transaction hợp lệ.

**When** validate bảng bằng transaction spec.

**Then** validation không được cast, thêm, xóa, đổi tên hoặc thay đổi giá trị
của bất kỳ cột nào.

## Các behavior chưa quyết định

Những mục dưới đây chưa được biến thành test cho đến khi contract được chốt:

- Extra column được phép hay bị reject?
- Null trong `entity_col` hoặc `time_col` thuộc schema hay data quality?
- Required column có phân biệt chữ hoa và chữ thường không?
- Một `BASKET_ID` thuộc nhiều `household_key` là data quality hay business rule?
- `time_grain = "1d"` chỉ là metadata hay cần validation?
- Schema có cần xác nhận semantic roles sau `resolve_schema()` không?

## Ngoài phạm vi hiện tại

- Kiểm tra datatype.
- Tạo hoặc validate target.
- Duplicate rows.
- Outlier và giá trị âm.
- Referential integrity giữa các bảng.
- Tự động sửa hoặc cast dữ liệu.

## Trạng thái

```text
Test plan: đang bổ sung
Executable tests: chưa viết
Implementation changes: chưa thực hiện từ tài liệu này
```
