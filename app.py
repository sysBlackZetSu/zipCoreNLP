# Import các thư viện cần thiết
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import matplotlib.pyplot as plt
import numpy as np

# Lấy dữ liệu từ dataset "tyqiangz/multilingual-sentiments" với ngôn ngữ là tiếng Nhật
dataset = load_dataset("tyqiangz/multilingual-sentiments", "japanese")

# Lấy tokenizer tự động cho mô hình BERT tiếng Nhật
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

# Lấy mô hình BERT tiếng Nhật cho bài toán phân loại chuỗi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = dataset["train"].features["label"].num_classes
model = (AutoModelForSequenceClassification
    .from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=num_labels)
    .to(device))

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)

# Create a DatasetDict
dataset_dict = DatasetDict({"train": dataset_encoded["train"], "validation": dataset_encoded["validation"]})

# Limit the training dataset
small_train_dataset = dataset_dict["train"].select([0, 1, 2, 3])

# Chuẩn bị tham số cho quá trình huấn luyện
batch_size = 16
logging_steps = len(small_train_dataset) // batch_size
model_name = f"sample-text-classification-distilbert"
training_args = TrainingArguments(
    output_dir=model_name, # Đường dẫn đến thư mục để lưu trữ các tệp liên quan đến quá trình đào tạo (mô hình, nhật ký, ...).
    num_train_epochs=2, # Số lượng epochs (vòng lặp qua toàn bộ tập dữ liệu) cho quá trình đào tạo.
    learning_rate=2e-5, # Tốc độ học của mô hình
    per_device_train_batch_size=batch_size, # Kích thước batch (số lượng mẫu đào tạo) trên mỗi thiết bị.
    per_device_eval_batch_size=batch_size, # Kích thước batch cho quá trình đánh giá (sử dụng khi kiểm thử hoặc đánh giá mô hình).
    weight_decay=0.01, # Hệ số giảm trọng lượng để tránh quá mức đào tạo.
    evaluation_strategy="epoch", # Xác định cách đánh giá mô hình sau mỗi epoch, có thể là "steps" hoặc "epoch".
    disable_tqdm=False, # Tắt thanh tiến trình
    logging_steps=500,  # Số lượng bước đào tạo giữa mỗi lần ghi log. Được sử dụng để giảm số lượng log nếu quá trình đào tạo quá dài.
    push_to_hub=False, # Đẩy mô hình và các tài nguyên liên quan lên Hugging Face Model Hub
    log_level="error", # Đặt mức độ log của quá trình đào tạo.
)
# Định nghĩa hàm tính các chỉ số đánh giá
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Huấn luyện mô hình
trainer = Trainer(
    model=model, args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=small_train_dataset,
    eval_dataset=dataset_encoded["validation"],
    tokenizer=tokenizer
)
trainer.train()

# Đánh giá mô hình trên tập validation và lấy dự đoán
preds_output = trainer.predict(dataset_encoded["validation"])

# Lấy nhãn dự đoán và nhãn thực tế
y_preds = np.argmax(preds_output.predictions, axis=1)
y_valid = np.array(dataset_encoded["validation"]["label"])
labels = dataset_encoded["train"].features["label"].names

# Hàm vẽ ma trận nhầm
def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

# Vẽ ma trận nhầm
plot_confusion_matrix(y_preds, y_valid, labels)

# Gán thông tin nhãn cho mô hình
id2label = {}
for i in range(dataset["train"].features["label"].num_classes):
    id2label[i] = dataset["train"].features["label"].int2str(i)

label2id = {}
for i in range(dataset["train"].features["label"].num_classes):
    label2id[dataset["train"].features["label"].int2str(i)] = i

trainer.model.config.id2label = id2label
trainer.model.config.label2id = label2id

# Lưu mô hình
trainer.save_model("/content/drive/MyDrive/sample-text-classification-bert")

# Load mô hình và tokenizer
new_tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/sample-text-classification-bert")
new_model = (AutoModelForSequenceClassification
    .from_pretrained("/content/drive/MyDrive/sample-text-classification-bert")
    .to(device))

# Dự đoán trên một đoạn văn bản mới
sample_text = "Your sample text here."
inputs = new_tokenizer(sample_text, return_tensors="pt")
new_model.eval()
with torch.no_grad():
    outputs = new_model(
        inputs["input_ids"].to(device), 
        inputs["attention_mask"].to(device),
    )
y_preds = np.argmax(outputs.logits.to('cpu').detach().numpy().copy(), axis=1)

# Hàm chuyển đổi ID nhãn thành nhãn tương ứng
def id2label(x):
    return new_model.config.id2label[x]
y_dash = [id2label(x) for x in y_preds]
y_dash
