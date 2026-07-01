import os
import modal
from pathlib import Path

app = modal.App("ail303m-fruit-classification")
cloud_volume = modal.Volume.from_name("ail303m-volume", create_if_missing=True)

@app.function(volumes={"/root/cloud_data": cloud_volume})
def get_cloud_files():
    # Liệt kê tất cả các file có trong ổ đĩa ảo
    try:
        files = os.listdir("/root/cloud_data")
    except FileNotFoundError:
        return {}
    
    file_data = {}
    for f in files:
        file_path = f"/root/cloud_data/{f}"
        if os.path.isfile(file_path):
            with open(file_path, "rb") as file:
                file_data[f] = file.read()
                
    return file_data

@app.local_entrypoint()
def main():
    print("[*] Đang kết nối tới Ổ đĩa vĩnh viễn trên Modal Cloud...")
    file_data = get_cloud_files.remote()
    
    if not file_data:
        print("[-] Không tìm thấy file nào trên Cloud Volume!")
        return

    local_project_dir = Path(__file__).resolve().parent.parent
    local_weights_dir = local_project_dir / "weights"
    local_output_dir = local_project_dir / "output"
    
    local_weights_dir.mkdir(exist_ok=True)
    local_output_dir.mkdir(exist_ok=True)
    
    for filename, content in file_data.items():
        if filename.endswith(".npy"):
            save_path = local_weights_dir / filename
            save_path.write_bytes(content)
            print(f"[+] Đã tải Trọng số (Weights): {save_path}")
        elif filename.endswith(".png"):
            save_path = local_output_dir / filename
            save_path.write_bytes(content)
            print(f"[+] Đã tải Biểu đồ (Plot): {save_path}")
        else:
            save_path = local_project_dir / filename
            save_path.write_bytes(content)
            print(f"[+] Đã tải File: {save_path}")
            
    print("[*] Hoàn tất quá trình tải file từ Cloud!")

if __name__ == "__main__":
    main()
