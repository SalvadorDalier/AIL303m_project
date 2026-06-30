# Nhật ký Thử nghiệm Huấn luyện (Experiment Log)

Tài liệu này lưu trữ lịch sử các lần chạy thử nghiệm huấn luyện mô hình, các tham số cấu hình, kết quả đạt được và các ghi chú quan trọng.

## Bảng Tóm tắt Thử nghiệm (Experiment Summary Table)

|STT|Ngày|Tên Mô hình|Batch Size|LR (Learning Rate)|Epochs|Optimizer|Loss (Train/Val)|Acc (Train/Val)|Ghi chú (Note)|
|-|-|-|-|-|-|-|-|-|-|
|1|2026-06-28|VGG16|36|0.001|5|Adam|0.26652 / 0.18486|92.8% / 93.2%|chạy lâu, độ chính xác cao nhất|
|2|2026-06-28|VGG19|36|0.001|5|Adam|0.24 / 0.187|80.64% / 92.8%|lâu hơn, chính xác thấp hơn vgg16|
|3|2026-06-29|Restnet50|36|0,001|5|Adam|0,40/0,40|82,13% / 83,1%|chạy nhanh hơn vgg16\&19, độ chính xác trên val thấp nhất|

\---

## Chi tiết từng lần chạy (Detailed Run Logs)

### Thử nghiệm #1: VGG16

* **Ngày:** 2026-06-28
* **Tham số chi tiết:**

  * Batch Size: 36
  * Learning Rate: 0.001
  * Epochs: 5
  * Optimizer: Adam
  * Loss Function: CrossEntropyLoss
  * Image Size: 224x224
* **Kết quả:**

  * Train Loss: 0.26 | Val Loss: 0.18
  * Train Acc: 92.8% | Val Acc: 93.2%
* **Ghi chú / Đánh giá:**

  * Mô hình chạy lâu, 1 batch mất gần 5 phút
  * độ chính xác cao nhất (có thể do data đơn giản)

### Thử nghiệm #2: VGG19

* **Ngày:** 2026-06-28
* **Tham số chi tiết:**

  * Batch Size: 5
  * Learning Rate: 0.001
  * Epochs: 5
  * Optimizer: Adam
  * Loss Function: CrossEntropyLoss
* **Kết quả:**

  * Train Loss: 0.24 | Val Loss: 0.18
  * Train Acc: 80.64% | Val Acc: 92.8%
* **Ghi chú / Đánh giá:**

  * Độ chính xác trên val không cải thiện nhiều, train acc thấp hơn so với vgg16 rất nhiều (có thể do epoch quá thấp)
  * Thời gian chạy lâu hơn vgg19

### Thử nghiệm #3: Restnet50

* **Ngày:** 2026-06-29
* **Tham số chi tiết:**

  * Batch Size: 5
  * Learning Rate: 0.001
  * Epochs: 5
  * Optimizer: Adam
  * Loss Function: CrossEntropyLoss
* **Kết quả:**

  * Train Loss: 0.4 | Val Loss: 0.4
  * Train Acc: 82.13% | Val Acc: 83.1%
* **Ghi chú / Đánh giá:**

  * Mô hình chạy nhanh hơn cả vgg16\&19
  * Train và val acc lại thấp hơn cả 2 vgg (có thể do thiếu data, chạy ít epoch)

