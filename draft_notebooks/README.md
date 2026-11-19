# Danh mục Notebooks (draft_notebooks)

Thư mục này chứa toàn bộ các Jupyter Notebook phục vụ quá trình khám phá dữ liệu (EDA), phân tích chuyên sâu (Deep Dive) theo loại biến, và đánh giá mối quan hệ giữa các nhóm đặc trưng với nhãn mục tiêu (*Purchase Propensity Label*).

---

## 1. Tổng quan & Deep Dive Dữ liệu Cơ sở (EDA & Semantic Deep Dive)

| STT | File Notebook | Tên / Chủ đề | Mục tiêu & Nội dung chính thực hiện |
| :---: | :--- | :--- | :--- |
| 1 | [`00_data_overview.ipynb`](00_data_overview.ipynb) | **Tổng quan Dữ liệu (Data Overview)** | • Thống kê kích thước bảng (rows, cols), memory usage và kiểu dữ liệu của 8 bảng Dunnhumby.<br>• Kiểm định Candidate Grain, tính duy nhất (`is_unique`) và tính hợp lệ.<br>• Phân tích missing rate, blank strings, null mở rộng và integrity khóa liên bảng.<br>• Đánh giá sơ bộ phân phối numeric, categorical và activity gap giữa các basket. |
| 2 | [`01_deep_dive_identifier.ipynb`](01_deep_dive_identifier.ipynb) | **Deep Dive — Identifier** | • Phân tích tần suất và mật độ xuất hiện của các khóa định danh (`household_key`, `BASKET_ID`, `STORE_ID`, `MANUFACTURER`).<br>• Tóm tắt số dòng, số basket, ngày active và span hoạt động theo từng household.<br>• Trực quan hóa phân phối log số dòng trên mỗi khóa định danh. |
| 3 | [`02_deep_dive_numeric.ipynb`](02_deep_dive_numeric.ipynb) | **Deep Dive — Numeric** | • Phân tích 5 biến định lượng: `QUANTITY`, `SALES_VALUE`, `RETAIL_DISC`, `COUPON_DISC`, `COUPON_MATCH_DISC`.<br>• Tính thống kê phân phối (mean, std, min, percentiles, skew, zero rate, âm/dương).<br>• Phân tích hành vi mua số lượng lớn (`QUANTITY` đuôi log-scale) và tách nhóm sản phẩm xăng.<br>• Xếp hạng tổng `QUANTITY` theo Department, Commodity và Department × Commodity. |
| 4 | [`03_deep_dive_temporal.ipynb`](03_deep_dive_temporal.ipynb) | **Deep Dive — Temporal** | • Khảo sát các biến thời gian: `DAY`, `WEEK_NO`, `TRANS_TIME`, `START_DAY`, `END_DAY`.<br>• Dashboard hoạt động basket theo ngày, giai đoạn và khung giờ mua sắm.<br>• Phân phối số tuần active và heatmap độ phủ (coverage) theo tuần của từng household.<br>• Đếm số basket theo thứ trong tuần × khung giờ giao dịch; tính thời lượng campaign. |
| 5 | [`04_deep_dive_Nominal.ipynb`](04_deep_dive_Nominal.ipynb) | **Deep Dive — Nominal** | • Phân tích danh mục sản phẩm: Top-N và Pareto số dòng giao dịch theo Department, Commodity, Sub-commodity.<br>• Đóng góp doanh số `SALES_VALUE` theo Department, Commodity (Community) và Department × Commodity.<br>• So sánh National Brand vs. Private Brand về catalog, doanh số và household reach.<br>• Phân tích Product long-tail (Pareto 80%) và Pareto Household theo Sales, Lines, Baskets. |
| 6 | [`06_outlier_analysis.ipynb`](06_outlier_analysis.ipynb) | **Outlier Analysis** | • Đánh giá phân phối dị biệt (outlier) đa chiều bằng Histogram Linear, Log-count và Boxplot.<br>• Tính ngưỡng IQR và đưa ra kết luận/đề xuất xử lý cho từng cột (giữ / cap / loại).<br>• Đánh giá lại outlier sau khi loại trừ giao dịch xăng (Gasoline vs. Non-gasoline).<br>• Preview danh sách top 100 dòng cực trị và tổng hợp sản phẩm extreme theo p99. |
| 7 | [`04_category_analysis.ipynb`](04_category_analysis.ipynb) | **Category Analysis (Placeholder)** | • Notebook dự phòng dành cho các phân tích danh mục chuyên biệt mở rộng. |
| 8 | [`05_customer_segmentation.ipynb`](05_customer_segmentation.ipynb) | **Customer Segmentation (Placeholder)** | • Notebook dự phòng dành cho phân khúc và phân cụm khách hàng. |

---

## 2. Phân tích Đặc trưng Hành vi theo Label (Label Analysis)

Tất cả các notebook trong nhóm này phân tích hành vi khách hàng trong cửa sổ quan sát **90 ngày trước thời điểm cutoff**, so sánh đối chứng giữa nhóm khách hàng tiềm năng (*Positive: label = 1*) và nhóm còn lại (*Negative: label = 0*).

| STT | File Notebook | Nhóm đặc trưng | Mục tiêu & Nội dung chính thực hiện |
| :---: | :--- | :--- | :--- |
| 9 | [`07_hh_demographic_label_analysis.ipynb`](07_hh_demographic_label_analysis.ipynb) | **Nhân khẩu học (Demographics)** | • So sánh quy mô, độ phủ và cơ cấu nhân khẩu học (độ tuổi, thu nhập, tình trạng hôn nhân, quy mô gia đình) giữa 2 nhóm label.<br>• Tính Positive Rate theo từng nhóm nhân khẩu học và kiểm tra độ lệch phân phối. |
| 10 | [`08_transaction_time_label_analysis.ipynb`](08_transaction_time_label_analysis.ipynb) | **Thời gian & Nhịp mua (Time & Recency)** | • So sánh số ngày từ lần mua gần nhất (*Recency / Days since last purchase*).<br>• Đánh giá số ngày active, số tuần active, khoảng cách giữa các lần mua (*interpurchase gap*).<br>• Vẽ boxplot phân bố thời gian đối sánh giữa 2 nhãn. |
| 11 | [`09_transaction_frequency_monetary_label_analysis.ipynb`](09_transaction_frequency_monetary_label_analysis.ipynb) | **Tần suất & Giá trị chi tiêu (Frequency & Monetary)** | • So sánh tổng chi tiêu (*Total Sales*), số lượng basket, chi tiêu trung bình/trung vị trên mỗi giỏ hàng.<br>• Đánh giá mức độ chi tiêu trên mỗi ngày active giữa khách hàng Positive và Negative. |
| 12 | [`10_transaction_quantity_basket_label_analysis.ipynb`](10_transaction_quantity_basket_label_analysis.ipynb) | **Số lượng & Quy mô giỏ hàng (Quantity & Basket)** | • So sánh tổng số lượng hàng mua (*Total Quantity*), số item trung bình trong mỗi basket.<br>• Phân tích độ đa dạng sản phẩm (số lượng unique product, department breadth) theo label. |
| 13 | [`11_product_category_label_analysis.ipynb`](11_product_category_label_analysis.ipynb) | **Lịch sử Ngành hàng (Category History)** | • Khảo sát tỷ lệ từng mua các Department, Commodity, Sub-commodity giữa 2 nhóm.<br>• Xác định các ngành hàng chủ lực (dominant category) có tỷ lệ khách hàng Positive vượt trội. |
| 14 | [`12_product_brand_manufacturer_label_analysis.ipynb`](12_product_brand_manufacturer_label_analysis.ipynb) | **Thương hiệu & Nhà sản xuất (Brand & Manufacturer)** | • So sánh lịch sử chọn National Brand vs. Private Brand theo label.<br>• Phân tích độ đa dạng thương hiệu, mức độ trung thành (loyalty concentration) và nhà sản xuất đặc trưng. |
| 15 | [`13_promotion_retail_discount_label_analysis.ipynb`](13_promotion_retail_discount_label_analysis.ipynb) | **Khuyến mại Bán lẻ (Retail Discount)** | • So sánh tỷ lệ khách hàng từng mua hàng giảm giá bán lẻ và positive rate tương ứng.<br>• Phân tích tỷ trọng giỏ hàng có chiết khấu và độ nhạy giá (*discount intensity* / tỷ lệ giảm giá trên sales). |
| 16 | [`14_coupon_label_analysis.ipynb`](14_coupon_label_analysis.ipynb) | **Mã giảm giá (Coupon)** | • Đánh giá tỷ lệ sử dụng coupon (*Ever redeemed*), số lần quy đổi coupon và positive rate.<br>• So sánh thời gian từ lần dùng coupon cuối cùng và tổng giá trị giảm trừ coupon (`COUPON_DISC`, `COUPON_MATCH_DISC`). |
| 17 | [`15_merchandising_label_analysis.ipynb`](15_merchandising_label_analysis.ipynb) | **Trưng bày & Tiếp thị (Display & Mailer)** | • Phân tích mức độ tiếp xúc với trưng bày tại điểm bán (*Display exposure*) và tờ rơi quảng cáo (*Mailer exposure*).<br>• Đánh giá tác động của tiếp thị tại cửa hàng lên tỷ lệ chuyển đổi Positive Label. |
| 18 | [`read_label.ipynb`](read_label.ipynb) | **Tiện ích Đọc Nhãn (Label Inspection)** | • Tải artifact label mục tiêu và kiểm tra nhanh schema, số lượng bản ghi và tỷ lệ phân bố nhãn. |
