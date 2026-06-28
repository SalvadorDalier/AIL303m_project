# Nhật ký Thử nghiệm Huấn luyện (Experiment Log)

Tài liệu này lưu trữ lịch sử các lần chạy thử nghiệm huấn luyện mô hình, các tham số cấu hình, kết quả đạt được và các ghi chú quan trọng.

## Bảng Tóm tắt Thử nghiệm (Experiment Summary Table)

| STT | Ngày | Tên Mô hình | Batch Size | LR (Learning Rate) | Epochs | Optimizer | Loss (Train/Val) | Acc (Train/Val) | Ghi chú (Note) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-06-28 | GoogLeNet | 32 | 0.001 | 20 | Adam | 0.25 / 0.32 | 92.5% / 89.1% | Lần chạy đầu tiên, mô hình hội tụ tốt. |
| 2 | 2026-06-28 | DenseNet-121 | 16 | 0.0001 | 30 | Adam | 0.18 / 0.24 | 95.2% / 91.8% | Kết quả tốt hơn GoogLeNet, nhưng huấn luyện lâu hơn. |
| 3 | | | | | | | | | |

---

## Chi tiết từng lần chạy (Detailed Run Logs)

### Thử nghiệm #1: GoogLeNet Baseline
- **Ngày:** 2026-06-28
- **Mô hình:** GoogLeNet (Inception v1)
- **Tham số chi tiết:**
  - Batch Size: 32
  - Learning Rate: 0.001 (không decay)
  - Epochs: 20
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss
  - Image Size: 224x224
- **Kết quả:**
  - Train Loss: 0.25 | Val Loss: 0.32
  - Train Acc: 92.5% | Val Acc: 89.1%
- **Ghi chú / Đánh giá:**
  - Mô hình chạy mượt mà, không gặp lỗi bộ nhớ (OOM).
  - Có dấu hiệu Overfitting nhẹ từ epoch 15 trở đi. Cần xem xét thêm dropout hoặc weight decay ở lần tiếp theo.

---

### Thử nghiệm #2: DenseNet-121
- **Ngày:** 2026-06-28
- **Mô hình:** DenseNet-121
- **Tham số chi tiết:**
  - Batch Size: 16
  - Learning Rate: 0.0001
  - Epochs: 30
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss
- **Kết quả:**
  - Train Loss: 0.18 | Val Loss: 0.24
  - Train Acc: 95.2% | Val Acc: 91.8%
- **Ghi chú / Đánh giá:**
  - Độ chính xác cải thiện rõ rệt so với GoogLeNet.
  - Thời gian train mỗi epoch lâu hơn khoảng 1.5 lần.
