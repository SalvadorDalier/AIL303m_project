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
  - Mọi hình ảnh đều được thay đổi kích thước (Resize) về chuẩn `224x224` pixel (tạo thành dạng tensor `[Batch, 3, 224, 224]`). Lý do rất đơn giản: Các mạng CNN mà nhóm kế thừa (như VGG, ResNet) vốn được thiết kế và huấn luyện trên tập ImageNet với đầu vào mặc định là 224x224. Nếu đưa ảnh sai kích thước, các phép nhân ma trận trọng số (đặc biệt ở các lớp Fully Connected) sẽ bị báo lỗi không khớp chiều (shape mismatch) ngay lập tức.
  - **Trọng tâm kỹ thuật:** Giải thích lý do phải chuẩn hóa (Normalize) theo `mean=[0.485, 0.456, 0.406]` và `std=[0.229, 0.224, 0.225]`. Đây không phải là con số ngẫu nhiên, mà là dải phân phối thống kê của hàng triệu bức ảnh từ ImageNet. Nếu bỏ qua bước này, không gian vector của ảnh đầu vào sẽ lệch pha với không gian trọng số pre-trained, làm sụp đổ phương pháp Transfer Learning.

## Slide 4: Chiến lược Học chuyển giao (Transfer Learning Architecture)
- **Nội dung hiển thị (Visual):** Sơ đồ kiến trúc mạng (Ví dụ: DenseNet/VGG), biểu tượng ổ khóa (Freeze) ở phần thân và biểu tượng mở khóa ở phần đuôi Classifier.
- **Phân tích chuyên sâu (In-depth):** 
  - Dự án khai thác 5 kiến trúc hàng đầu: GoogLeNet, DenseNet-121, ResNet-50, VGG-16, VGG-19.
  - **Cơ chế hoạt động:** Sử dụng kỹ thuật "Đóng băng" (`requires_grad = False`) cho toàn bộ phần Feature Extractor. Các Convolutional Layers nông đã quá giỏi trong việc nhận diện góc cạnh, và các layer sâu đã quen với pattern phức tạp. Việc đóng băng giúp tránh hiện tượng "Catastrophic Forgetting" (Quên thảm họa) khi train trên miền dữ liệu mới.
  - Sau đó, nhóm tự tinh chỉnh và huấn luyện duy nhất cụm Fully Connected Layer (Classifier) cuối cùng để quy mô hình về bài toán 2 nơ-ron (Healthy/Unhealthy).

## Slide 5: Hàm Tối ưu & Hàm Mất mát (Optimization & Loss Logic)
- **Nội dung hiển thị (Visual):** Công thức CrossEntropy, tham số thuật toán Adam Optimizer (`lr=1e-3` / `1e-4`).
- **Phân tích chuyên sâu (In-depth):** *(Đây là phần bắt buộc phải hiểu sâu để bảo vệ đồ án)*
  - Khẳng định bài toán **không sử dụng hàm Sigmoid**. Sigmoid chỉ phù hợp với phân loại đa nhãn độc lập (Multi-label).
  - Hàm **`CrossEntropyLoss`** trong PyTorch là sự kết hợp toán học trực tiếp của **`LogSoftmax`** và **`NLLLoss`** (Negative Log Likelihood Loss). Softmax đảm bảo tổng xác suất của 2 nơ-ron đầu ra luôn bằng 1 (tạo thành một phân phối xác suất chuẩn), và NLLLoss sẽ tối ưu hóa khoảng cách Kullback-Leibler giữa dự đoán và nhãn thực tế.
  - Ở khâu Inference (Dự đoán), để phân giải chốt hạ ảnh thuộc nhãn nào, mô hình sử dụng hàm **`torch.max()` (Argmax)** để lấy index của nơ-ron có Logit lớn nhất.

## Slide 6: Triển khai Huấn luyện Đám mây (Cloud Computing via Modal)
- **Nội dung hiển thị (Visual):** Sơ đồ quy trình đẩy luồng huấn luyện lên nền tảng Modal, kiến trúc Cloud Volume.
- **Phân tích chuyên sâu (In-depth):** 
  - Trình bày kỹ năng MLOps (Machine Learning Operations). Thay vì train cục bộ gây quá tải phần cứng, job được đóng gói và đẩy lên máy chủ đám mây.
  - Điểm sáng: Tích hợp cơ chế **Persistent Volume** (Ổ cứng ảo). Giải quyết triệt để rủi ro mất kết nối mạng (Network Timeout) trong quá trình train hàng chục epochs kéo dài nhiều giờ; file trọng số (Weights) được lưu an toàn trên mây và kéo về bằng script độc lập.

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
