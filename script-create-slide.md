# Script Thuyết trình Đồ án: Phân loại Trái cây (AIL303m)
*Kịch bản dành cho 2 người thuyết trình (Dự kiến 10-15 phút)*

---

## PHẦN 1: TỔNG QUAN & PHƯƠNG PHÁP LÕI (NGƯỜI A TRÌNH BÀY)

### Slide 1: Giới thiệu Đề tài (Title)
- **Nội dung:** Tên đề tài, Tên thành viên, Giảng viên hướng dẫn.

### Slide 2: Đặt vấn đề & Khó khăn dữ liệu
- **Nội dung:** Bài toán phân loại trái cây Khỏe (Healthy) và Bệnh (Unhealthy).
- **Điểm nhấn:** Khó khăn lớn nhất của học sâu là thiếu dữ liệu. Nhóm đã áp dụng phương pháp tiên tiến từ bài báo nghiên cứu: Dùng **CGAN (Conditional GAN)** để sinh thêm dữ liệu giả lập (synthetic data) mô phỏng các vết nấm mốc (mould) và thối rữa (gangrene). (Dataset nâng lên 2152 Train / 538 Valid).

### Slide 3: Định nghĩa Input & Output
- **Input:** Ảnh RGB, tiền xử lý Resize về `224x224` pixel, chuẩn hóa (Normalize) theo tập ImageNet.
- **Output:** 2 Nơ-ron (Logits) tương ứng 2 nhãn 0 (Healthy) và 1 (Unhealthy). Khi ứng dụng thực tế sẽ đi qua hàm **Softmax** để xuất ra % xác suất.

### Slide 4: Kỹ thuật Trích xuất Đặc trưng (Transfer Learning)
- Trình bày 3 bước xây dựng mô hình:
  1. Kế thừa kiến trúc sâu (DenseNet, VGG, ResNet) đã pre-train trên ImageNet.
  2. **Đóng băng (Freeze)** phần thân (`requires_grad = False`) để tiết kiệm chi phí tính toán, giữ lại khả năng trích xuất hình khối, vân màu cực tốt của AI.
  3. Cắt bỏ lớp xuất cũ, **thay và train lại lớp Classifier mới** để chuyên biệt hóa cho trái cây.

### Slide 5: Hàm Mất mát & Hàm Ra Quyết định (Câu hỏi bẫy của Hội đồng)
- **Loss Function (Hàm Mất mát lúc huấn luyện):** Dùng `CrossEntropyLoss`. 
- **⚠️ Đính chính kiến thức:** Nếu bị vặn hỏi *"Hàm này có dùng Sigmoid không?"*. Bạn A dõng dạc trả lời: *"Dạ không, CrossEntropyLoss trong PyTorch tích hợp **LogSoftmax + NLLLoss**. Do cấu trúc xuất ra 2 nơ-ron nên nhóm dùng Softmax để tổng 2 xác suất bằng 100%, chứ không dùng Sigmoid (dành cho 1 nơ-ron / Multi-label) ạ!"*.
- **Decision Function (Hàm chốt hạ kết quả 0 hay 1):** Để máy tính thực sự quyết định ảnh thuộc nhãn `0 (Healthy)` hay `1 (Unhealthy)`, dự án dùng hàm **`torch.max()`** (hay còn gọi là hàm *Argmax*). Hàm này đơn giản là so sánh điểm số (Logit) hoặc % xác suất của cả 2 nơ-ron, nơ-ron nào có điểm cao hơn thì mô hình sẽ phán quyết thuộc class đó.

---

## PHẦN 2: ĐÁNH GIÁ, KẾT QUẢ & XAI (NGƯỜI B TRÌNH BÀY)

### Slide 6: Kết quả Huấn luyện trên Cloud
- Nhóm đã cấu hình đưa việc training lên nền tảng đám mây **Modal**. Vừa tiết kiệm tài nguyên máy cá nhân, vừa tích hợp cơ chế Cloud Volume (Két sắt trên mây) để giữ an toàn cho file trọng số (Weights) nếu bị rớt mạng.

### Slide 7: Bảng Đánh giá 5 Mô hình (Confusion Matrix & Recall)
- **Nội dung:** Đưa bảng LaTeX chứa chỉ số Precision & Recall của 5 mô hình vào đây.
- **Phân tích của Người B:** 
  - **DenseNet-121:** Làm việc khắt khe nhất, khả năng tóm trái cây thối đỉnh nhất (Recall Unhealthy = 89.03%). Thà giết lầm không bỏ sót.
  - **VGG-16 & VGG-19:** Rất bao dung, khả năng khẳng định trái cây khỏe xuất sắc (Recall Healthy > 94%). 

### Slide 8: Khám phá thú vị: Hiện tượng Overconfidence
- **Nội dung:** Đưa ảnh chụp màn hình % Softmax của VGG (báo 100% và 0%).
- **Kịch bản:** *"Khi chạy thực tế, VGG thường xuất ra xác suất tuyệt đối 100%. Đây không phải lỗi code, mà là hiện tượng **Overconfidence (Quá tự tin)** rất đặc trưng của các mạng Neural sâu như VGG khi fine-tune. Trọng lượng mạng lớn đẩy hàm Softmax về các thái cực tuyệt đối."*

### Slide 9: "Mắt AI" - Explainable AI (Grad-CAM)
- **Nội dung:** Trình diễn ảnh Heatmap đỏ/xanh từ file `explain.py`.
- **Kịch bản:** *"Để chứng minh AI không học vẹt, nhóm dùng Grad-CAM nội soi mô hình. Vùng màu Đỏ trên màn hình minh chứng AI đang chú ý chính xác vào vết nấm mốc để đưa ra quyết định Unhealthy."*

### Slide 10: Demo Ứng dụng Thực tế (`predict.py`)
- Show video hoặc ảnh chụp Terminal chạy lệnh `predict.py`, in ra tỷ lệ % cho một bức ảnh cụ thể. Minh chứng sản phẩm hoàn thiện từ A-Z.

### Slide 11: Kết luận & Hướng phát triển
- Chốt lại thành quả đạt được và gửi lời cảm ơn Hội đồng. Dành thời gian cho Q&A.
