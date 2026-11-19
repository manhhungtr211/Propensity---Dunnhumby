## Note
Draf_notebooks là file nháp để đào sâu eda về các khía cạnh, chưa file phải bản final

# Propensity VSF

Phân tích dữ liệu bán lẻ Dunnhumby và xây dựng dữ liệu cho bài toán dự đoán khả năng một household sẽ tiếp tục mua hàng trong tương lai gần.

## Bài toán

Đơn vị phân tích chính là một household tại một thời điểm chốt (`cutoff_day`). Dữ liệu lịch sử trước thời điểm chốt được dùng để tạo feature; giao dịch xảy ra sau thời điểm chốt được dùng để tạo target.


## Dữ liệu nguồn

Tám bảng Dunnhumby nằm trong `data/raw/Dunnhumby`:

- `transaction_data.csv`: dòng sản phẩm trong basket, household, thời gian, quantity, sales và discount.
- `product.csv`: taxonomy sản phẩm, brand và manufacturer.
- `hh_demographic.csv`: đặc điểm demographic của household.
- `campaign_desc.csv`: thời gian và loại campaign.
- `campaign_table.csv`: household được gán vào campaign.
- `coupon.csv`: coupon, product và campaign.
- `coupon_redempt.csv`: lịch sử redeem coupon.
- `causal_data.csv`: display/mailer theo product, store và week.

## Luồng notebook

### Khám phá dữ liệu

- `00_data_overview.ipynb`: tổng quan structure, schema, grain, quality và coverage.
- `01_deep_dive_identifier.ipynb`: tần suất và phân phối identifier.
- `02_deep_dive_numeric.ipynb`: numeric diagnostics, QUANTITY và taxonomy liên quan.
- `03_deep_dive_temporal.ipynb`: hoạt động theo ngày, tuần, giờ và campaign.
- `04_deep_dive_Nominal.ipynb`: category, brand, product long-tail và Pareto household.
- `06_outlier_analysis.ipynb`: histogram, boxplot và outlier; tách GASOLINE/NON_GAS.

`04_category_analysis.ipynb` và `05_customer_segmentation.ipynb` hiện là notebook placeholder.

### Phân tích feature theo label

- `07_hh_demographic_label_analysis.ipynb`: demographic.
- `08_transaction_time_label_analysis.ipynb`: recency và temporal behavior.
- `09_transaction_frequency_monetary_label_analysis.ipynb`: frequency và monetary.
- `10_transaction_quantity_basket_label_analysis.ipynb`: quantity và basket.
- `11_product_category_label_analysis.ipynb`: department, commodity và subcommodity.
- `12_product_brand_manufacturer_label_analysis.ipynb`: brand và manufacturer.
- `13_promotion_retail_discount_label_analysis.ipynb`: retail discount.
- `14_coupon_label_analysis.ipynb`: coupon behavior.
- `15_merchandising_label_analysis.ipynb`: display và mailer exposure.

## Dataset đã xử lý

Dataset T30/H7/step14 nằm tại `data/processed/purchase_propensity_t30_h7_step14`:

- `household_cutoff_features.parquet`: feature và target leakage-safe.
- `feature_target_associations.parquet`: association feature–target theo cutoff.
- `feature_correlation_matrix.csv`: ma trận tương quan đã tổng hợp theo thời gian.

Chi tiết grain, time policy và feature nằm trong README của thư mục processed tương ứng.

## Cài đặt

Yêu cầu Python 3.11 trở lên.

```powershell
uv sync --extra viz --extra dev
```

Chạy test:

```powershell
uv run pytest
```

Build lại feature dataset:

```powershell
uv run python scripts/build_purchase_feature_dataset.py
```

## Nguyên tắc phân tích

- Không dùng dữ liệu sau `cutoff_day` để tạo feature.
- Luôn phân biệt số dòng transaction, basket distinct và household distinct.
- Outlier thống kê không mặc định là lỗi nghiệp vụ; `QUANTITY` xăng phải được phân tích riêng.
- Correlation và chênh lệch label chỉ mang ý nghĩa mô tả, không phải quan hệ nhân quả.
