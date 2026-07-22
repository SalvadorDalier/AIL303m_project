# CẤU TRÚC THUYẾT TRÌNH CHUYÊN SÂU: PHÂN LOẠI CHẤT LƯỢNG TRÁI CÂY (AIL303m)
*(Tài liệu này cung cấp luận điểm logic và chiều sâu kỹ thuật cho từng Slide, dùng làm tài liệu nền tảng để thiết kế PowerPoint và bảo vệ trước Hội đồng)*

---

## Slide 1: Đặt vấn đề & Thách thức cốt lõi (Problem Statement)
- **Nội dung hiển thị (Visual):** Tên đề tài, hình ảnh minh họa trái cây khỏe và trái cây bị mốc/thối, kèm theo biểu đồ bánh (Pie Chart) chênh lệch số lượng mẫu thực tế.
- **Phân tích chuyên sâu (In-depth):** 
  - Khẳng định đây không chỉ là bài toán phân loại nhị phân thông thường, mà là bài toán **thị giác máy tính vi mô** (nhận diện các khiếm khuyết nhỏ như vết nấm mốc - mould, hay vết thối rữa - gangrene trên bề mặt).
  - **Chứng minh sự mất cân bằng dữ liệu (Data Imbalance):**
    - *Về mặt thực tế:* Trong nông nghiệp, việc thu thập ảnh trái cây khỏe mạnh (Healthy) cực kỳ dễ dàng (chỉ cần ra vườn chụp hàng ngàn tấm). Nhưng để săn được ảnh trái cây bị mốc/thối tự nhiên là rất hiếm và khó kiểm soát.
    - *Hậu quả lên AI:* Nếu mang một dataset chênh lệch (VD: 1000 ảnh Khỏe - 100 ảnh Bệnh) đi train, AI sẽ bị hội chứng "lười biếng" (Bias). Nó chỉ cần nhắm mắt phán tất cả là "Khỏe" thì độ chính xác tổng vẫn đạt 90%. Hậu quả là nó sẽ mù lòa hoàn toàn trước trái cây bệnh.
    - *Lật vấn đề:* Đây chính là lý do dự án bắt buộc phải tìm đến công nghệ sinh ảnh giả lập (CGAN) ở Slide tiếp theo để cứu vãn sự mất cân bằng này!

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

## Slide 8: Giải thích Mô hình với Explainable AI (Giải quyết bài toán Hộp đen)
- **Nội dung hiển thị (Visual):** Các bức ảnh biểu đồ nhiệt (Heatmap) xuất ra từ thuật toán Grad-CAM.
- **Phân tích chuyên sâu (In-depth):** 
  - **Vấn đề Hộp đen (Black-box):** Từ trước đến nay, mạng CNN thường bị coi là "hộp đen". Khi nó phán 1 quả táo là "Unhealthy", con người không thể biết tại sao nó lại phán như vậy. Nó thực sự nhìn thấy vết nấm mốc, hay nó chỉ nhìn nhầm vào một cái bóng mờ ở background?
  - **Giải pháp của nhóm (Grad-CAM):** Để chứng minh mô hình không hề "học vẹt", nhóm đã tích hợp thuật toán **Grad-CAM**. Thuật toán này hoạt động như một chiếc máy chụp X-quang, quét ngược từ kết quả cuối cùng về lại các lớp Tích chập (Convolution) để xem khu vực nào có trọng số ảnh hưởng cao nhất.
  - **Kết luận:** Vùng màu Đỏ trên Heatmap thể hiện sự chú ý cao nhất của AI. Các biểu đồ nhiệt minh chứng rõ ràng rằng AI của nhóm đã thực sự học được cách focus ánh nhìn trực tiếp vào các vết nấm mốc / thối rữa trên bề mặt trái cây để ra quyết định.

## Slide 9: Xây dựng Pipeline Dự đoán Thực tế (End-to-End Inference)
- **Nội dung hiển thị (Visual):** Demo kiến trúc file `predict.py`, giao diện dòng lệnh trả kết quả %.
- **Phân tích chuyên sâu (In-depth):** 
  - Trình bày vòng đời cuối cùng của sản phẩm: Từ 1 bức ảnh thô -> Đi qua toàn bộ ma trận trọng số (Weights `.pth` / `.npy`) đã train -> Xuất ra Logits thô.
  - Áp dụng hàm `F.softmax` tại bước cuối cùng để phiên dịch Logits thành xác suất phần trăm, cung cấp một kết quả trực quan, dễ hiểu và sẵn sàng tích hợp vào phần mềm cho người nông dân sử dụng.

## Slide 10: Tổng kết & Hướng phát triển tương lai
- **Nội dung hiển thị (Visual):** Tóm tắt 3 điểm mạnh nhất của dự án và các bước tiếp theo.
- **Phân tích chuyên sâu (In-depth):** 
  - Tổng kết: Hoàn thiện thành công một Pipeline ML hiện đại kết hợp CGAN (Tạo dữ liệu), Transfer Learning (Huấn luyện tối ưu) và XAI (Giải thích minh bạch).
  - Hướng phát triển: Đưa mô hình lên API/Web-app hoặc nén mô hình để chạy trên thiết bị di động (Edge Devices) ngoài thực địa.
