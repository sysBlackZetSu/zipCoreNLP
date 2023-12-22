# Load Dữ Liệu:
Sử dụng hàm load_dataset từ thư viện datasets để tải dữ liệu từ tập dữ liệu "tyqiangz/multilingual-sentiments" với ngôn ngữ là tiếng Nhật.
Sử dụng AutoTokenizer để tạo tokenizer cho mô hình BERT tiếng Nhật.
Sử dụng AutoModelForSequenceClassification để tạo mô hình BERT tiếng Nhật cho bài toán phân loại chuỗi.

# Tokenize Dữ Liệu:
Sử dụng hàm tokenize để áp dụng tokenizer cho tất cả các batch trong tập dữ liệu.
Tạo một đối tượng DatasetDict để lưu trữ các tập dữ liệu "train" và "validation".

# Giới Hạn Dữ Liệu Huấn Luyện:
Sử dụng hàm select từ datasets để giới hạn tập dữ liệu huấn luyện thành một phần nhỏ.

# Chuẩn Bị Tham Số Cho Quá Trình Huấn Luyện:
Định nghĩa các tham số như đường dẫn đầu ra, số epochs, tốc độ học, kích thước batch, và các tham số khác cho quá trình đào tạo.
Tạo một đối tượng TrainingArguments để chứa các tham số đào tạo.

# Huấn Luyện Mô Hình:
Sử dụng Trainer từ thư viện transformers để huấn luyện mô hình.
Cung cấp hàm tính chỉ số đánh giá và tập dữ liệu huấn luyện đã giới hạn.
Thực hiện quá trình đào tạo và lưu mô hình.

# Đánh Giá Mô Hình:
Sử dụng mô hình đã đào tạo để dự đoán trên tập dữ liệu validation.
Tính toán các chỉ số đánh giá như độ chính xác và F1 score.
Vẽ ma trận nhầm để hiển thị hiệu suất của mô hình.
Gán Nhãn Cho Mô Hình:

Gán thông tin nhãn cho mô hình bằng cách xây dựng các ánh xạ từ ID nhãn sang nhãn và ngược lại.

# Lưu và Tải Mô Hình:
Lưu mô hình và tokenizer vào đường dẫn được chỉ định.
Tải lại mô hình và tokenizer từ đường dẫn đã lưu.
Dự Đoán Trên Văn Bản Mới:

Sử dụng mô hình và tokenizer đã tải để dự đoán trên một đoạn văn bản mới.


Ref data test: 
- https://github.com/tyqiangz/multilingual-sentiment-datasets/blob/main/data
- https://huggingface.co/datasets/tyqiangz/multilingual-sentiments
