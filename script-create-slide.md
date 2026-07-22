# Script Thuyết trình Đồ án: Phân loại Trái cây (Fruit Quality Classification)

## Slide 1: Giới thiệu & Định nghĩa Dữ liệu (Input & Output)
- **Input của Model:**
  - Ảnh RGB (3 kênh màu), được tiền xử lý Resize về kích thước chuẩn `224x224` pixel (chuẩn của các kiến trúc CNN hiện đại).
  - Ảnh được chuẩn hóa (Normalize) theo mean/std của ImageNet để đồng bộ với Transfer Learning: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- **Output của Model:**
  - Là 2 nơ-ron xuất (2 logits) tương ứng với 2 nhãn phân loại nhị phân:
    - `0`: Healthy (Trái cây khỏe mạnh/bình thường)
    - `1`: Unhealthy (Trái cây bị bệnh, mốc, thối - mould/gangrene)
  - Ở khâu ứng dụng thực tế (Inference/Predict), Logits này được ép qua hàm Softmax để tính ra **% xác suất** (VD: Healthy = 95%, Unhealthy = 5%).

## Slide 2: Kiến trúc & Hàm Tối ưu (Optimizer)
- **Hàm tối ưu (Optimized Function):**
  - Toàn bộ các mô hình (GoogLeNet, DenseNet, ResNet, VGG) đều được huấn luyện bằng thuật toán **Adam Optimizer**.
  - Tốc độ học (Learning Rate): `0.001` (hoặc `0.0001` tùy model), giúp mô hình hội tụ nhanh và tránh bị kẹt ở cực tiểu địa phương.

- **Hàm mất mát (Loss Function - Đính chính kiến thức):**
  - Trong dự án, ta dùng **`CrossEntropyLoss`**.
  - *Lưu ý cực kỳ quan trọng khi bị Hội đồng phản biện:* `CrossEntropyLoss` trong PyTorch **KHÔNG** tự động thêm hàm `Sigmoid`! Thay vào đó, nó tự động tích hợp **`LogSoftmax` + `NLLLoss`** (Negative Log Likelihood Loss) ngay bên trong hàm. 
  - Đó là lý do tại sao ở file Train ta không cần gọi hàm Softmax, nhưng khi đem đi Test (`predict.py`) thì ta phải tự thêm Softmax vào để dịch ra phần trăm (%). *(Nếu dùng Sigmoid thì đó là hàm `BCEWithLogitsLoss` dùng cho multi-label, không phải hàm đang dùng trong dự án này).*

## Slide 3: Kỹ thuật Trích xuất Đặc trưng (Feature Extraction)
Feature Extraction trong dự án này được thiết kế theo cơ chế **Transfer Learning (Học chuyển giao)**, cụ thể theo 3 bước:
1. **Sử dụng sức mạnh tiền nhân:** Kế thừa các mô hình kiến trúc sâu (DenseNet, VGG, ResNet...) đã được huấn luyện sẵn (Pre-trained) trên kho dữ liệu khổng lồ ImageNet.
2. **Đóng băng (Freeze) lớp trích xuất:** Khóa toàn bộ tham số của phần thân mô hình (Feature Extractor) bằng lệnh `requires_grad = False`. Việc này giúp ta tận dụng được khả năng nhận diện hình khối, góc cạnh, vân màu rất sắc sảo của AI có sẵn mà không tốn công/thời gian dạy lại từ đầu.
3. **Thay mới và Huấn luyện lớp phân loại (Classifier):** Cắt bỏ lớp dự đoán 1000 class cũ của ImageNet, thay bằng một cụm mạng Neural mới (gồm Linear, ReLU, Dropout, và lớp xuất ra 2 class). Quá trình Training thực chất chỉ là việc "dạy" cho cái đuôi mô hình này cách xâu chuỗi các đặc trưng đã được trích xuất để chốt hạ xem trái cây bị bệnh hay khỏe.

---
### 💡 Đề xuất thêm (Gợi ý từ tôi để Slide của bạn "Ăn điểm" cao hơn):
Nếu có đủ thời lượng và không gian thuyết trình, bạn cân nhắc đưa thêm 2 Slide này vào nhé (rất ăn tiền học thuật):
1. **Slide 4: Xử lý Hạn chế Dữ liệu bằng CGAN:** Nói về việc dùng *Conditional GAN* để sinh thêm dữ liệu ảnh giả lập (synthetic data), bù đắp sự thiếu hụt ảnh trái cây bệnh. Hãy nhấn mạnh đây là pipeline (quy trình) tiên tiến được chứng minh từ bài báo gốc.
2. **Slide 5: Mô phỏng "Mắt AI" bằng XAI (Grad-CAM):** Trình diễn các hình ảnh Heatmap đỏ-xanh xuất ra từ file `explain.py`. Giải thích cho Hội đồng rằng: *"Nhóm chúng em không chỉ làm ra AI phân loại vẹt, mà còn buộc nó phải chỉ ra chính xác vết nấm mốc nằm ở đâu trên trái cây bằng Grad-CAM!"*. Khúc này đảm bảo Hội đồng sẽ rất ấn tượng.
