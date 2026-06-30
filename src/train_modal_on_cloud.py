import os
import sys
from pathlib import Path
import modal

# 1. Định nghĩa ứng dụng Modal
app = modal.App("ail303m-fruit-classification")

# 2. Định nghĩa Container Image (Môi trường chạy trên Cloud)
# Cài đặt PyTorch hỗ trợ GPU, torchvision và các thư viện cần thiết
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", 
        "torchvision", 
        "tqdm", 
        "matplotlib",
        "pillow",
        "torchinfo"
    )
)

# 3. Định vị thư mục dự án cục bộ để tải lên Cloud
local_project_dir = Path(__file__).resolve().parent.parent

# Cấu hình tải toàn bộ thư mục dự án lên Cloud, loại bỏ các thư mục rác/nặng
image = image.add_local_dir(
    local_path=local_project_dir,
    remote_path="/root/project",
    ignore=["**/.venv", "**/.git", "**/__pycache__", "**/weights", "**/output"]
)

# 4. Hàm huấn luyện chạy trên GPU của Modal Cloud
@app.function(
    image=image,
    gpu="A10G",          # Sử dụng GPU NVIDIA A10G (24GB VRAM) hiệu năng cao
    timeout=21600        # Tăng giới hạn thời gian lên 6 tiếng (21600s) để chạy đủ 1000 epochs
)
def train_on_cloud(model_name: str, lr: float, epoch: int, batch: int, stop: int):
    print("=" * 60)
    print(f"[*] KHỞI ĐẦU HUẤN LUYỆN TRÊN MODAL CLOUD")
    print(f"[*] Model: {model_name.upper()} | LR: {lr} | Epochs: {epoch} | Stop: {stop} | Batch Size: {batch}")
    print("=" * 60)

    # Chuyển môi trường làm việc vào thư mục src trên Cloud
    os.chdir("/root/project/src")
    sys.path.append("/root/project/src")

    # Import các thành phần từ pipeline gốc của dự án
    from train import train_model
    import argparse

    # Tạo class giả lập Argument Parser để truyền vào train_model
    args = argparse.Namespace(
        model=model_name,
        lr=lr,
        epoch=epoch,
        stop=stop,
        worker=2,
        batch=batch,
        verbose=1 # In kết quả gọn gàng sau mỗi epoch
    )

    # Thực thi quá trình train
    train_model(args)

    # Đọc dữ liệu của file Trọng số (.npy) và Biểu đồ (.png) đã sinh ra trên Cloud
    if model_name == "googlenet22":
        weight_file = "googlenet.npy"
        curve_file = "GOOGLENET22_Test_curves.png"
    elif model_name == "densenet121":
        weight_file = "densenet.npy"
        curve_file = "DENSENET121_Test_curves.png"
    else:
        weight_file = f"{model_name}.npy"
        curve_file = f"{model_name.upper()}_Test_curves.png"

    cloud_weights_path = Path(f"/root/project/weights/{weight_file}")
    cloud_curve_path = Path(f"/root/project/output/{curve_file}")

    # Đọc file nhị phân để trả về máy khách (local)
    weights_bytes = cloud_weights_path.read_bytes() if cloud_weights_path.exists() else None
    curve_bytes = cloud_curve_path.read_bytes() if cloud_curve_path.exists() else None

    print("[+] Quá trình huấn luyện trên Cloud hoàn tất!")
    return weights_bytes, curve_bytes, weight_file, curve_file


# 5. Điểm chạy chính ở máy cục bộ (Local Client Entrypoint)
@app.local_entrypoint()
def main(model: str = "googlenet22", lr: float = 0.001, epoch: int = 2000, batch: int = 32, stop: int = 30):
    """
    Chạy lệnh: modal run src/train_modal.py --model googlenet22 --epoch 2000 --stop 30
    """
    # Gửi lệnh lên Cloud chạy và đợi kết quả trả về
    weights_bytes, curve_bytes, weight_file, curve_file = train_on_cloud.remote(
        model_name=model, 
        lr=lr, 
        epoch=epoch, 
        batch=batch,
        stop=stop
    )

    # Nhận và lưu lại file Trọng số về máy local
    if weights_bytes:
        local_weights_dir = local_project_dir / "weights"
        local_weights_dir.mkdir(exist_ok=True)
        local_weights_path = local_weights_dir / weight_file
        local_weights_path.write_bytes(weights_bytes)
        print(f"[+] Đã tải trọng số từ Cloud về máy cục bộ:\n    -> {local_weights_path}")
    else:
        print("[-] Không tìm thấy file trọng số được sinh ra trên Cloud.")

    # Nhận và lưu lại file Biểu đồ về máy local
    if curve_bytes:
        local_output_dir = local_project_dir / "output"
        local_output_dir.mkdir(exist_ok=True)
        local_curve_path = local_output_dir / curve_file
        local_curve_path.write_bytes(curve_bytes)
        print(f"[+] Đã tải biểu đồ từ Cloud về máy cục bộ:\n    -> {local_curve_path}")
    else:
        print("[-] Không tìm thấy file biểu đồ được sinh ra trên Cloud.")
