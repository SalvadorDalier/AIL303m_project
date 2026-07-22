# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT: PHÂN LOẠI CHẤT LƯỢNG TRÁI CÂY (AIL303m)
*(Lưu ý: Đây là kịch bản thoại chi tiết (word-by-word) dành cho 2 người thuyết trình. Các bạn có thể điều chỉnh lại văn phong cho thuận miệng)*

---

## 🎙️ PHẦN 1: MỞ ĐẦU & PHƯƠNG PHÁP CỐT LÕI (NGƯỜI A TRÌNH BÀY)

### Slide 1: Chào hỏi & Giới thiệu đề tài
**[Người A nói]:** 
"Dạ kính chào quý thầy cô trong Hội đồng và các bạn. Hôm nay, nhóm chúng em xin phép được trình bày đồ án với đề tài: **Phân loại Trái cây Khỏe mạnh và Sâu bệnh (Fruit Quality Classification)**. 
Bài toán đặt ra là làm sao xây dựng được một mô hình AI có khả năng tự động soi chiếu và phát hiện các khiếm khuyết của trái cây như nấm mốc (mould) hay thối rữa (gangrene)."

### Slide 2: Khó khăn Dữ liệu & Giải pháp CGAN
**[Người A nói]:**
"Thưa thầy cô, khó khăn lớn nhất khi làm AI trong nông nghiệp là sự thiếu hụt dữ liệu ảnh bệnh thực tế. Để giải quyết, nhóm em đã tham khảo một bài báo nghiên cứu tiên tiến và ứng dụng **CGAN (Conditional GAN)**.
Thay vì đi thu thập thêm ảnh thật, hệ thống CGAN này đã giúp nhóm sinh ra các dữ liệu ảnh trái cây sâu bệnh giả lập (synthetic data) có độ chân thực rất cao. Nhờ vậy, bộ dataset của nhóm đã được cân bằng và mở rộng lên mức lý tưởng là **2152 ảnh Training và 538 ảnh Validation**."

### Slide 3: Quy chuẩn Input / Output & Hàm Ra Quyết định
**[Người A nói]:**
"Về thiết kế cốt lõi của hệ thống:
- **Đầu vào (Input):** Mọi bức ảnh đưa vào đều được tiền xử lý Resize về chuẩn `224x224` pixel, và chuẩn hóa (Normalize) theo dải màu của tập ImageNet.
- **Đầu ra (Output):** Mô hình xuất ra 2 nơ-ron điểm số tương ứng với 2 nhãn: `0 (Healthy)` và `1 (Unhealthy)`.
- **Cách AI chốt kết quả:** Để máy tính quyết định ảnh thuộc nhãn 0 hay nhãn 1, nhóm em sử dụng hàm **`torch.max()`** (hay còn gọi là hàm Argmax). Hàm này làm một phép toán rất cơ bản là so sánh điểm số của nơ-ron 0 và nơ-ron 1, bên nào điểm cao hơn thì mô hình sẽ phán quyết ảnh thuộc class đó."

### Slide 4: Chiến lược Transfer Learning (Học chuyển giao)
**[Người A nói]:**
"Để tối ưu hóa thời gian huấn luyện, nhóm em không xây mạng AI từ con số 0. Chúng em sử dụng **Transfer Learning** trên 5 kiến trúc đình đám: GoogLeNet, DenseNet-121, ResNet-50, VGG-16 và VGG-19. 
Chiến lược của nhóm gồm 2 bước: 
Thứ nhất, **Đóng băng (Freeze)** toàn bộ phần thân mô hình (`requires_grad = False`). Việc này giúp AI tận dụng được con mắt nhìn hình khối đã quá giỏi của ImageNet mà không tốn công học lại. 
Thứ hai, nhóm chỉ cắt bỏ chiếc đuôi cũ và **train lại một cụm Classifier mới** để chuyên biệt hóa cho việc phân biệt bệnh của trái cây."

### Slide 5: Tối ưu & Hàm Mất mát (Chống phản biện)
**[Người A nói]:**
"Trong quá trình Train, nhóm em dùng thuật toán **Adam** với Learning rate là `0.001` (hoặc `0.0001` tùy mạng) để mô hình hội tụ nhanh nhất. Hàm mất mát được sử dụng là **CrossEntropyLoss**.

*(Nếu Hội đồng ngắt lời hỏi: Hàm này phân loại 2 nhãn thì có dùng Sigmoid không?)*
-> **[Người A đáp ngay]:** Dạ thưa thầy/cô không ạ. Hàm `CrossEntropyLoss` trong PyTorch không hề tích hợp Sigmoid. Nó tích hợp hàm **LogSoftmax** kết hợp với **NLLLoss**. Vì mạng của nhóm em xuất ra 2 nơ-ron nên tụi em dùng nguyên lý Softmax để tổng 2 xác suất bằng 100%, chứ không dùng Sigmoid (BCE Loss) như mô hình 1 nơ-ron ạ!"

---
*(Người A nhường lời cho Người B)*
---

## 🎙️ PHẦN 2: THỰC NGHIỆM, XAI & KẾT LUẬN (NGƯỜI B TRÌNH BÀY)

### Slide 6: Kết quả Huấn luyện (Train trên Cloud)
**[Người B nói]:**
"Cảm ơn bạn A. Tiếp theo, em xin trình bày về phần thực nghiệm. Nhằm đối phó với việc khối lượng tính toán quá lớn, nhóm em đã đưa toàn bộ code lên nền tảng đám mây **Modal**. Việc này giúp giải phóng GPU máy cá nhân, đồng thời nhóm đã lập trình một cơ chế *Két sắt ảo (Cloud Volume)* để nếu mạng có bị đứt giữa chừng thì trọng số (Weights) vẫn được bảo toàn và tải về an toàn."

### Slide 7: So sánh 5 Mô hình (Confusion Matrix & Recall)
**[Người B nói]:**
"Trên màn hình là bảng đối chiếu chỉ số **Precision** và **Recall** của 5 mô hình trên tập Validation (538 ảnh).
Nhìn vào bảng, ta thấy 2 thái cực hoàn toàn khác nhau:
- **Nhà vô địch khắt khe - DenseNet-121:** Mô hình này tóm trái cây bệnh giỏi nhất, đạt Recall Unhealthy lên tới **89.03%**. Tức là nó thà giết lầm chứ không bỏ sót, rất an toàn để chặn hàng hỏng ra thị trường.
- **Nhà vô địch bao dung - VGG:** VGG-16 và VGG-19 lại xuất sắc trong việc khẳng định trái cây khỏe (Recall Healthy đạt mức khổng lồ trên **94%**). Rất ít trái ngon nào bị VGG phân loại oan."

### Slide 8: Phân tích hiện tượng Overconfidence
**[Người B nói]:**
"Có một hiện tượng thú vị ở mô hình VGG là khi xuất % xác suất, nó thường xuyên trả về mức tuyệt đối là `100.00%` và `0.00%`. Nhóm em đã nghiên cứu và phát hiện đây là hiện tượng **Overconfidence (Quá tự tin)** rất đặc trưng của Deep Learning. Do kiến trúc VGG quá sâu, các tham số điểm số (Logits) trở nên cực kỳ lớn, đẩy hàm Softmax lọt vào các thái cực tiệm cận 1 và 0. Khẳng định đây không phải là lỗi rò rỉ dữ liệu (data leakage) mà là bản chất của kiến trúc!"

### Slide 9: Giải thích AI (Explainable AI - Grad-CAM)
**[Người B nói]:**
"Để chứng minh cho Hội đồng thấy mô hình không hề học vẹt, nhóm em đã tích hợp thuật toán **Grad-CAM**. Trên màn hình là biểu đồ Nhiệt (Heatmap). Vùng màu đỏ chính là vị trí mà mạng Nơ-ron đang tập trung sự chú ý. 
Như thầy cô thấy, con AI đã 'nhìn' trúng phóc vào vết nấm mốc để đưa ra quyết định Unhealthy. *(Thêm vào nếu muốn: Trong quá trình làm XAI, nhóm em đã khắc phục thành công rào cản In-place ReLU của DenseNet và lỗi Gradient biến mất do Freeze layers để sinh ra được bức ảnh này ạ).* "

### Slide 10: Ứng dụng Thực tế (Sản phẩm đầu cuối)
**[Người B nói]:**
"Cuối cùng, nhóm không chỉ dừng lại ở chỉ số lý thuyết. Nhóm đã đóng gói một kịch bản dự đoán `predict.py`. Khi đưa một bức ảnh bất kỳ ở đời thực vào, mô hình sẽ tự động trích xuất điểm số, sau đó áp dụng toán học **Softmax** để dịch ra tỷ lệ % trực quan cho người nông dân (Ví dụ: Khỏe mạnh 98%, Sâu bệnh 2%)."

### Slide 11: Lời Kết & Q&A
**[Người B nói]:**
"Tóm lại, đồ án đã chứng minh việc kết hợp Transfer Learning, CGAN Data và Explainable AI mang lại một quy trình kiểm định chất lượng nông sản cực kỳ mạnh mẽ và minh bạch. 
Phần trình bày của nhóm đến đây là kết thúc. Chúng em xin cảm ơn quý thầy cô đã lắng nghe và rất mong nhận được những góp ý, câu hỏi từ Hội đồng ạ!"
