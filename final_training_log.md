# Nhật ký Huấn luyện Tổng hợp (Final Training Log) - AIL303m Project

Tài liệu này là bản tổng hợp cuối cùng toàn bộ quá trình huấn luyện của 5 kiến trúc mô hình trong dự án Phân loại chất lượng trái cây (Healthy vs Unhealthy). Dữ liệu được tổng hợp từ các file log rời rạc trước đó.

## 1. Thông số Cấu hình Chung (Global Configurations)
- **Dataset:** 2152 ảnh Train / 538 ảnh Valid (Có sử dụng kỹ thuật CGAN để sinh thêm dữ liệu bù đắp sự thiếu hụt).
- **Phân loại nhãn (Classes):** `0: Healthy`, `1: Unhealthy` (Mould/Gangrene).
- **Hàm Mất mát (Loss Function):** CrossEntropyLoss.
- **Thuật toán Tối ưu (Optimizer):** Adam.
- **Kích thước ảnh đầu vào (Image Size):** 224x224.
- **Môi trường Huấn luyện:** Nền tảng Đám mây (Modal Cloud Platform) + persistent Cloud Volume để sao lưu trọng số.

---

## 2. Bảng Tóm tắt Kết quả Huấn luyện (Summary Table)

| STT | Tên Mô hình | Batch Size | Learning Rate | Epochs (Dự kiến/Dừng) | Train Acc | Val Acc | Đánh giá Sơ bộ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **GoogLeNet** | 32 | 0.001 | 20 / - | 92.5% | **89.1%** | Baseline chạy mượt, có dấu hiệu overfit nhẹ sau epoch 15. (*Cloud log ghi nhận Val Acc 73.79% ở một phiên chạy khác*). |
| 2 | **DenseNet-121** | 16 | 0.0001 | 1000 / 83 | 95.2% | **91.8%** | Mô hình ổn định nhất, đặc trưng sâu. Train lâu hơn 1.5 lần. Bắt trái cây bệnh (Recall Unhealthy) tốt nhất. |
| 3 | **ResNet-50** | 5 | 0.001 | 5 / - | 82.13% | **83.1%** | Tốc độ train rất nhanh, nhưng độ chính xác chưa cao do số lượng Epochs thiết lập quá ít. |
| 4 | **VGG-16** | 36 | 0.001 | 5 / - | 92.8% | **93.2%** | Thời gian train cực lâu (gần 5 phút/batch). Độ chính xác trên giấy cao, cực kỳ bao dung (Recall Healthy rất tốt). |
| 5 | **VGG-19** | 5 | 0.001 | 5 / - | 80.64% | **92.8%** | Chậm hơn VGG-16. Train Acc thấp hơn hẳn nhưng Val Acc vẫn neo cao (do hiện tượng Overconfidence). |

---

## 3. Chi tiết Cập nhật & Sự cố Đáng chú ý (Incident & Update Logs)

- **Sự cố Early Stopping trên Cloud (Tháng 7/2026):**
  - Trong một phiên chạy dự kiến 1000 epochs, nhóm đặt lệnh stop tự động ở epoch 83 (do nhận thấy loss không giảm thêm). Tuy nhiên do lỗi thiết lập, máy chủ không tự ngắt gây tốn kém credit. 
  - Lệnh train mẫu: `run .\train_modal_on_cloud.py --model densenet121 --lr 1e-4 --epoch 1000 --stop 83`

- **Sự cố Explainable AI (Lỗi `NoneType` trong Grad-CAM):**
  - **Mô tả lỗi:** Khi xuất biểu đồ nhiệt bằng `explain.py`, hệ thống báo lỗi `TypeError: mean() received an invalid combination of arguments - got (NoneType, dim=tuple...)`.
  - **Nguyên nhân:** Do chiến lược Transfer Learning đã gán `requires_grad = False` (đóng băng trọng số) cho phần thân mô hình. Khi hàm backward() chạy ngược, nó bỏ qua tính Gradient cho các layer này, khiến biến `self.gradient` trả về rỗng (`None`).
  - **Khắc phục:** Mở khóa lại bằng lệnh `for param in model.parameters(): param.requires_grad = True` ngay trước khi chạy thuật toán Grad-CAM để lừa hệ thống ép tính Gradient. Đã fix thành công!

- **Lưu ý Đặc thù Kỹ thuật (Overconfidence):**
  - Hai mô hình họ VGG thường xuyên trả về tỷ lệ dự đoán tuyệt đối (`100%` và `0%`) khi đưa qua Softmax. Đây là đặc tính cấu trúc mạng quá sâu khiến Logits khuếch đại lớn, không phải lỗi rò rỉ dữ liệu (Data Leakage).

---
*(Tài liệu này đã được chuẩn hóa để đính kèm vào báo cáo hoặc khóa luận tốt nghiệp).*
