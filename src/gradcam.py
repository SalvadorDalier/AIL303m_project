import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse

# Gọi file chức năng từ thư mục module_going để thiết lập thư viện
from module_going.env_setup import setup_refer_lib
project_root = setup_refer_lib()

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError as e:
    print(f"Lỗi khi import thư viện pytorch-grad-cam: {e}")
    print("Vui lòng kiểm tra lại thư mục refer/pytorch-grad-cam")
    sys.exit(1)

from model.googlenet22 import googlenet_model
from model.densenet121 import densenet_model

def get_target_layer(model, model_name):
    # Trả về Target layer (layer cuối cùng trước khi Flatten) để Grad-CAM tính toán
    if model_name == "googlenet22":
        return [model.inception5b]
    elif model_name == "densenet121":
        return [model.features.norm5]
    else:
        raise ValueError("Không xác định được target layer cho model này.")

def main():
    parser = argparse.ArgumentParser(description="Chạy Grad-CAM cho ảnh truyền vào bằng thư viện jacobgil/pytorch-grad-cam")
    parser.add_argument('--model', type=str, default='googlenet22', choices=['googlenet22', 'densenet121'])
    parser.add_argument('--img', type=str, required=True, help='Đường dẫn tới ảnh cần test')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Khởi tạo model
    print(f"Khởi tạo mô hình {args.model.upper()}...")
    if args.model == 'googlenet22':
        model = googlenet_model(num_class=2)
        weight_file = 'googlenet.npy' # Tên file weight trong thư mục weights
    else:
        model = densenet_model(num_class=2)
        weight_file = 'densenet.npy' # Tên file weight trong thư mục weights (giả định)
        
    # Nạp weights mặc định từ folder weights
    weight_path = Path(project_root) / 'weights' / weight_file
    print(f"Đang nạp trọng số từ: {weight_path}")
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict.get('state_dict', state_dict))
    print("Đã nạp trọng số thành công!")
            
    model = model.eval().to(device)
    target_layers = get_target_layer(model, args.model)
    
    # 2. Khởi tạo thư viện GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 3. Load và tiền xử lý ảnh
    try:
        img = Image.open(args.img).convert('RGB')
    except Exception as e:
        print(f"Lỗi khi đọc file ảnh '{args.img}': {e}")
        return

    # Ảnh float32 scale [0,1] để vẽ overlay heatmap
    rgb_img = np.float32(img.resize((224, 224))) / 255
    
    # Transform tensor chuẩn để model xử lý
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # 4. Chạy Grad-CAM
    print("Đang chạy thuật toán Grad-CAM...")
    # targets=None mặc định sẽ lấy class có xác suất cao nhất mà mô hình dự đoán ra
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # 5. Phủ màu Heatmap lên ảnh gốc bằng hàm có sẵn của thư viện
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Hiển thị và lưu
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title('Ảnh gốc')
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    out_name = f"../gradcam_output_{args.model}.png"
    plt.tight_layout()
    plt.savefig(out_name)
    print(f"Thành công! Đã lưu ảnh kết quả tại {out_name}")
    plt.show()

if __name__ == '__main__':
    main()
