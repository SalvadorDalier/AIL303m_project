# -*- coding: utf-8 -*-
"""
File: C:\\Users\\Lenovo\\Desktop\\AIL303m_project\\src\\explain.py
Chức năng: Thực hiện Explainable AI (XAI) bằng thuật toán Grad-CAM tự lập trình (không dùng thư viện ngoài)
           để giải thích quyết định phân loại bệnh trái cây của mô hình GoogLeNet.
Tác giả: Chuyên gia PyTorch & Kỹ sư Computer Vision
"""

import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from torchvision import transforms

#
# c/
sys.path.append(str(Path(__file__).resolve().parent))
from model.googlenet22 import googlenet_model
from model.densenet121 import densenet_model
from model.resnet50 import resnet50_model
from model.vgg16 import vgg16_model
from model.vgg19 import vgg19_model


class GradCAM:
    """
    Lớp triển khai thuật toán Grad-CAM (Gradient-weighted Class Activation Mapping)
    sử dụng cơ chế Hook của PyTorch để bắt Feature Map và Gradients.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        # Đăng ký hook thu thập dữ liệu
        self._register_hooks()

    def _register_hooks(self):
        """
        Đăng ký Forward Hook và Backward Hook vào layer mục tiêu.
        """
        def forward_hook(module, input, output):
            # Lưu trữ feature map ở luồng forward
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Lưu trữ gradients ở luồng backward
            self.gradients = grad_output[0].detach()

        # Đăng ký hook vào layer mục tiêu
        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor, target_class=None):
        """
        Tính toán bản đồ kích hoạt Grad-CAM.
        """
        # 1. Lan truyền xuôi (Forward Pass)
        output = self.model(input_tensor)
        
        # Nếu không chỉ định class, lấy class có xác suất cao nhất (Argmax)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Xóa các gradients cũ
        self.model.zero_grad()
        
        # 2. Lan truyền ngược (Backward Pass) từ score của class mục tiêu
        loss = output[0, target_class]
        loss.backward()

        # 3. Tính toán trọng số alpha (Global Average Pooling của Gradients)
        # gradients shape: [1, C, H, W] -> alpha shape: [1, C, 1, 1]
        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. Nhân trọng số với Feature Map và tính tổng chập
        # activations shape: [1, C, H, W] -> cam shape: [H, W]
        cam = torch.sum(alpha * self.activations, dim=1).squeeze(0)

        # 5. Áp dụng hàm ReLU để chỉ giữ lại các đặc trưng có đóng góp dương
        cam = F.relu(cam)

        # 6. Chuẩn hóa ma trận CAM về khoảng [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        return cam.cpu().numpy(), target_class

    def remove_hooks(self):
        """
        Gỡ bỏ các hook để giải phóng bộ nhớ.
        """
        for handler in self.handlers:
            handler.remove()


def preprocess_image(image_path):
    """
    Đọc ảnh bằng OpenCV và áp dụng tiền xử lý đồng nhất với dataset.py.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

    # Đọc ảnh dạng BGR
    img = cv2.imread(image_path)
    
    # Resize ảnh về kích thước chuẩn 224x224 bằng nội suy Bilinear
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Chuyển đổi BGR sang RGB cho PyTorch
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Định nghĩa phép tiền xử lý tương đương validation/test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Thực hiện transform và thêm chiều batch: [C, H, W] -> [1, C, H, W]
    input_tensor = transform(img_rgb).unsqueeze(0)
    
    return img_resized, input_tensor


def main():
    # Cấu hình đối số dòng lệnh
    parser = argparse.ArgumentParser(description="Chạy Explainable AI (Grad-CAM) giải thích mô hình phân loại trái cây.")
    parser.add_argument('--model', type=str, default='googlenet22', choices=['googlenet22', 'densenet121', 'resnet50', 'vgg16', 'vgg19'],
                        help="Chọn mô hình để trực quan hóa (googlenet22, densenet121, resnet50, vgg16, vgg19).")
    parser.add_argument('--img', type=str, required=True,
                        help="Đường dẫn tới file ảnh cần giải thích.")
    parser.add_argument('--weights', type=str, default=None,
                        help="Đường dẫn cụ thể tới file weights. Nếu để trống sẽ tự lấy trong thư mục weights gốc.")
    args = parser.parse_args()

    # Thiết lập thư mục và thiết bị chạy
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Đang sử dụng thiết bị: {device}")

    # Định nghĩa nhãn đầu ra
    class_names = {0: "Healthy", 1: "Unhealthy"}

    # 1. Khởi tạo kiến trúc mô hình và xác định target layer
    print(f"[*] Đang khởi tạo mô hình: {args.model.upper()}")
    if args.model == 'googlenet22':
        model = googlenet_model(num_class=2)
        target_layer = model.inception5b
        default_weight_name = 'googlenet.npy'
    elif args.model == 'densenet121':
        model = densenet_model(num_class=2)
        target_layer = model.features.norm5
        default_weight_name = 'densenet.npy'
    elif args.model == 'resnet50':
        model = resnet50_model(num_class=2)
        target_layer = model.layer4[-1]
        default_weight_name = 'resnet50.npy'
    elif args.model == 'vgg16':
        model = vgg16_model(num_class=2)
        target_layer = model.features[-1] # VGG16 features
        default_weight_name = 'vgg16.npy'
    elif args.model == 'vgg19':
        model = vgg19_model(num_class=2)
        target_layer = model.features[-1]
        default_weight_name = 'vgg19.npy'

    # 2. Định vị và nạp weights (.npy hoặc .pth)
    weight_path = args.weights
    if not weight_path:
        weight_path = project_root / 'weights' / default_weight_name

    print(f"[*] Đang nạp trọng số mô hình từ: {weight_path}")
    if not Path(weight_path).exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số tại {weight_path}. Hãy chạy train.py trước!")

    # Nạp file trọng số
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
    # Hỗ trợ lấy state_dict nếu nó được bọc trong dict lớn
    model.load_state_dict(state_dict.get('state_dict', state_dict))
    
    model = model.to(device)
    model.eval()

    # BẮT BUỘC: Bật lại tính toán Gradient cho tất cả các lớp
    # (Vì trong file định nghĩa model đã bị đóng băng bằng requires_grad = False)
    for param in model.parameters():
        param.requires_grad = True

    # 3. Tiền xử lý ảnh đầu vào
    img_resized, input_tensor = preprocess_image(args.img)
    input_tensor = input_tensor.to(device)

    # 4. Trích xuất Grad-CAM
    print("[*] Đang tính toán bản đồ đặc trưng Grad-CAM...")
    grad_cam = GradCAM(model, target_layer)
    cam_matrix, pred_class = grad_cam.generate_cam(input_tensor)
    grad_cam.remove_hooks() # Giải phóng hooks

    # 5. Xử lý ảnh bằng OpenCV để tạo Heatmap trực quan
    # Chuẩn hóa về [0, 255] và chuyển sang định dạng uint8
    heatmap_gray = np.uint8(255 * cam_matrix)
    
    # Phóng to ma trận heatmap lên 224x224 bằng nội suy Bilinear
    heatmap_resized = cv2.resize(heatmap_gray, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Áp dụng bản đồ màu COLORMAP_JET (Đỏ là kích hoạt mạnh, xanh là yếu)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Trộn ảnh heatmap với ảnh gốc (tỷ lệ 40% heatmap + 60% ảnh gốc)
    overlay_img = cv2.addWeighted(heatmap_color, 0.4, img_resized, 0.6, 0)

    # 6. Lưu kết quả ra thư mục output
    output_dir = project_root / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_label = class_names[pred_class]
    img_name = Path(args.img).stem
    output_path = output_dir / f"gradcam_{args.model}_{img_name}_pred_{pred_label}.png"

    # Lưu ảnh kết quả
    cv2.imwrite(str(output_path), overlay_img)
    
    print("="*60)
    print(f"[+] Dự đoán của mô hình: {pred_label} (Class ID: {pred_class})")
    print(f"[+] Kết quả Grad-CAM đã được lưu thành công tại:\n    -> {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
