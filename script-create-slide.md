# CẤU TRÚC THUYẾT TRÌNH CHUYÊN SÂU: PHÂN LOẠI CHẤT LƯỢNG TRÁI CÂY (AIL303m)
*(Tài liệu này cung cấp luận điểm logic và chiều sâu kỹ thuật cho từng Slide, dùng làm tài liệu nền tảng để thiết kế PowerPoint và bảo vệ trước Hội đồng)*

---

## Slide 1: Đặt vấn đề & Thách thức cốt lõi (Problem Statement)
- **Nội dung hiển thị (Visual):** Tên đề tài, hình ảnh minh họa trái cây khỏe và trái cây bị mốc/thối.
- **Phân tích chuyên sâu (In-depth):** 
  - Khẳng định đây không chỉ là bài toán phân loại nhị phân thông thường, mà là bài toán **thị giác máy tính vi mô** (nhận diện các khiếm khuyết nhỏ như vết nấm mốc - mould, hay vết thối rữa - gangrene trên bề mặt).
  - Nhấn mạnh thách thức lớn nhất trong AI nông nghiệp: "Data Imbalance" (Mất cân bằng dữ liệu) và sự thiếu hụt nghiêm trọng các hình ảnh mẫu bệnh thực tế chất lượng cao.

## Slide 2: Giải pháp Dữ liệu - Đột phá với CGAN
- **Nội dung hiển thị (Visual):** Sơ đồ Pipeline tạo data giả lập, biểu đồ số lượng Dataset (2152 Train / 538 Valid).
- **Phân tích chuyên sâu (In-depth):** 
  - Thay vì dùng các kỹ thuật Augmentation cơ bản (xoay, lật, chỉnh sáng - vốn không tạo ra tri thức mới), dự án áp dụng **CGAN (Conditional Generative Adversarial Networks)** dựa trên nghiên cứu học thuật tiên tiến.
- Model CGAN đã giúp nhóm giải quyết được vấn đề 'data imbalance' trong mảng phân loại nông sản. Bằng cách tạo ra những bức ảnh nhân tạo giúp giải quyết được vấn đề thiếu hụt dữ liệu trong lĩnh vực này. 

## Slide 3: Chuẩn hóa Đầu vào (Data Preprocessing Pipeline)
- **Nội dung hiển thị (Visual):** Dòng chảy xử lý ảnh: Raw Image -> Resize (224x224) -> Tensor -> Normalization.
- **Phân tích chuyên sâu (In-depth):** 
  - Mọi hình ảnh phải bị ép về kích thước tensor chuẩn `[B, 3, 224, 224]` để phù hợp với kiến trúc của các mạng CNN mà nhóm đang dùng bao gôm Densenet121, Googlenet22, Resnet50, VGG16-19.
  - **Trọng tâm kỹ thuật:** `mean=[0.485, 0.456, 0.406]` và `std=[0.229, 0.224, 0.225]`. Đây không phải là con số ngẫu nhiên, mà là dải phân phối thống kê của hàng triệu bức ảnh từ ImageNet được tính theo công thức: x_new = (x_old - mean)/std trên từng không gian màu [R-B-G]
## Slide 4: Chiến lược Học chuyển giao (Transfer Learning Architecture)
- **Nội dung hiển thị (Visual):** Sơ đồ kiến trúc mạng (Ví dụ: DenseNet/VGG), biểu tượng ổ khóa (Freeze) ở phần thân và biểu tượng mở khóa ở phần đuôi Classifier.
- **Phân tích chuyên sâu (In-depth):** 
  - Dự án khai thác 5 kiến trúc hàng đầu: GoogLeNet, DenseNet-121, ResNet-50, VGG-16, VGG-19.
  - **Cơ chế hoạt động:** Sử dụng kỹ thuật "Đóng băng" (`requires_grad = False`) cho toàn bộ phần Feature Extractor. Các lớp Convolution (Tích chập) này vốn đã quá giỏi trong việc nhận diện góc cạnh, vân màu. Việc đóng băng ma trận là thực sự cần thiết
  - nếu không, những sai số (Loss) khổng lồ ở các epoch đầu tiên sẽ dội ngược lại (backpropagation) và phá hỏng toàn bộ trí nhớ mà mô hình đã cất công học từ tập ImageNet.
  - Sau đó, nhóm tự tinh chỉnh và huấn luyện duy nhất cụm Fully Connected Layer (Classifier) cuối cùng để quy mô hình về bài toán 2 classes (Healthy/Unhealthy).

## Slide 5: Hàm Tối ưu & Hàm Mất mát (Optimization & Loss Logic)
- **Nội dung hiển thị (Visual):** Công thức CrossEntropy, tham số thuật toán Adam Optimizer (`lr=1e-3`)
  - Khẳng định bài toán **không sử dụng hàm Sigmoid**. Sigmoid chỉ phù hợp với phân loại đa nhãn độc lập (Multi-label).
  - Hàm **`CrossEntropyLoss`** trong PyTorch là sự kết hợp của 2 hàm **`LogSoftmax`** và **`NLLLoss`** (Negative Log Likelihood Loss). Softmax đảm bảo tổng xác suất của 2 nơ-ron đầu ra luôn bằng 1 (tạo thành một phân phối xác suất chuẩn). Còn NLLLoss sẽ đóng vai trò phạt trọng số: Nếu máy dự đoán sai mà lại đưa ra % tự tin càng cao thì điểm phạt (Loss) trả về sẽ càng lớn, ép mô hình phải điều chỉnh lại trọng số cho đúng.
  - Ở khâu prediction, để phân giải chốt hạ ảnh thuộc nhãn nào, mô hình sử dụng hàm **`torch.max()` (Argmax)** để lấy index của nơ-ron có số Logit lớn nhất.
  ps: số logit là output của lớp cuối cùng fc ép ma trận lại thành 1 dãy số dài.

## Slide 6: Chiến lược Huấn luyện Lai (Local RTX 4060 & Colab Tesla T4)
- **Nội dung hiển thị (Visual):** Bảng so sánh thông số phần cứng (Local RTX 4060 8GB vs Colab Tesla T4 15GB VRAM).
- **Phân tích chuyên sâu (In-depth):** 
  - Nhóm áp dụng chiến lược huấn luyện phân tán linh hoạt (Hybrid) để tối ưu hóa nút thắt cổ chai về phần cứng. 3 mô hình (GoogLeNet, DenseNet, ResNet) được train trực tiếp trên máy Local, tận dụng nhân CUDA của RTX 4060 để đẩy nhanh tốc độ thực thi.
  - Tuy nhiên, VGG-16 và VGG-19 là 2 kiến trúc tuyến tính được nhóm cho chạy trên T4-GPU của Tesla T4 

## Slide 7: Đánh giá Hiệu năng & Đặc tính Mô hình (Evaluation Metrics)
- **Nội dung hiển thị (Visual):** Bảng tổng hợp Precision, Recall và Confusion Matrix của 5 mô hình.
- **Phân tích chuyên sâu (In-depth):** Đưa ra góc nhìn mổ xẻ sự khác biệt về kiến trúc dẫn đến hiệu năng khác nhau:
  - **DenseNet-121:** Là mạng có khả năng nối chéo (Skip-connections) cực kỳ dày đặc, giúp "tái sử dụng đặc trưng" (Feature Reuse) tối đa. Nhờ vậy, nó cực kỳ nhạy cảm với các vết bệnh vi mô, đạt **Recall Unhealthy cao nhất (89.03%)**. Phù hợp làm màng lọc khắt khe trong kiểm định.
  - **VGG-16 & VGG-19:** Kiến trúc tuyến tính sâu, qua nhiều tầng Max Pooling làm phẳng dữ liệu khiến nó bao dung hơn. Hệ quả là nó cực kỳ xuất sắc trong việc bảo vệ trái cây khỏe (**Recall Healthy > 94%**), hiếm khi phán oan trái ngon.

## Slide 8: Hiện tượng Dữ liệu học thuật (Overconfidence Calibration)
- **Nội dung hiển thị (Visual):** Log Output của VGG in ra xác suất tuyệt đối `100.00%` và `0.00%`.
- **Phân tích chuyên sâu (In-depth):** 
  - Giải đáp thắc mắc về con số tuyệt đối: Đây không phải là Data Leakage (Rò rỉ dữ liệu). Đây là hiện tượng **Overconfidence** (Lệch chuẩn tự tin) kinh điển trong Deep Learning.
  - Lý do toán học: Do mạng VGG quá sâu, biên độ trọng số ở lớp Fully Connected cuối cùng rất lớn. Khi các điểm số (Logits) chênh lệch nhau lớn và đi qua hàm Exponential của Softmax, sự chênh lệch bị khuếch đại vô hạn, đẩy % xác suất lọt vào các thái cực tuyệt đối 1 và 0. 

## Slide 9: Minh bạch hóa Hộp đen AI (Explainable AI - XAI)
- **Nội dung hiển thị (Visual):** Các bức ảnh biểu đồ nhiệt (Heatmap) nội soi từ thuật toán Grad-CAM.
- **Phân tích chuyên sâu (In-depth):** 
  - Giải quyết bài toán "Black-box" (Hộp đen) của AI. Nhóm ứng dụng **Grad-CAM (Gradient-weighted Class Activation Mapping)**.
  - Cơ chế: Bằng cách gắn Hook vào các Convolutional Layer cuối cùng, tính đạo hàm ngược (Backward Gradients) để lấy trọng số không gian, sau đó nhân chập với Feature Maps.
  - Kết quả: Chứng minh rõ ràng trước Hội đồng rằng AI thực sự học được cách "nhìn" vào các vết nấm mốc (vùng màu đỏ trên Heatmap) để ra quyết định, chứ không hề học vẹt phần background.

## Slide 10: Xây dựng Pipeline Dự đoán Thực tế (End-to-End Inference)
- **Nội dung hiển thị (Visual):** Demo kiến trúc file `predict.py`, giao diện dòng lệnh trả kết quả %.
- **Phân tích chuyên sâu (In-depth):** 
  - Trình bày vòng đời cuối cùng của sản phẩm: Từ 1 bức ảnh thô -> Đi qua toàn bộ ma trận trọng số (Weights `.pth` / `.npy`) đã train -> Xuất ra Logits thô.
  - Áp dụng toán học `F.softmax` tại bước cuối cùng để phiên dịch Logits thành xác suất phần trăm, cung cấp một kết quả trực quan, dễ hiểu và sẵn sàng tích hợp vào phần mềm cho người nông dân sử dụng.

## Slide 11: Tổng kết & Hướng phát triển tương lai
- **Nội dung hiển thị (Visual):** Tóm tắt 3 điểm mạnh nhất của dự án và các bước tiếp theo.
- **Phân tích chuyên sâu (In-depth):** 
  - Tổng kết: Hoàn thiện thành công một Pipeline ML hiện đại kết hợp CGAN (Tạo dữ liệu), Transfer Learning (Huấn luyện tối ưu) và XAI (Giải thích minh bạch).
  - Hướng phát triển: Đưa mô hình lên API/Web-app hoặc nén mô hình (Pruning/Quantization) để chạy trên thiết bị di động (Edge Devices) ngoài thực địa.
