import torch
import gc
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model.googlenet22 import googlenet_model
from model.densenet121 import densenet_model
from model.resnet50 import resnet50_model
from model.vgg16 import vgg16_model
from model.vgg19 import vgg19_model
from dataset import dataloader
from utils import plot_curves

# implement far more models
def get_model(name, num_class=2):
    if name == "googlenet22": return googlenet_model(num_class)
    if name == "densenet121": return densenet_model(num_class)
    if name == "resnet50": return resnet50_model(num_class)
    if name == "vgg16": return vgg16_model(num_class)
    if name == "vgg19": return vgg19_model(num_class)
    else: raise ValueError(f'không có model: {name}')

# setup run models 
def train_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    
    # run dataloader
    train_loader, valid_loader = dataloader(batch_size=args.batch, num_workers=args.worker)

    print(f"Đang chạy model: {args.model.upper()}")
    model = get_model(args.model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    # --------------------------------------------------------
    # Cấu hình Early Stopping
    # --------------------------------------------------------
    best_val_loss = float('inf')
    stop = args.stop
    stop_counter = 0
    best_model_state = None

    for epoch in range(1, args.epoch + 1):
        if args.verbose == 2:
            print(f"Epoch {epoch}/{args.epoch}")
    # Vòng lặp huấn luyện chính
    try:
        for epoch in range(args.epoch):
            print(f"\n[+] Epoch {epoch+1}/{args.epoch}")
            
            # --- TRAIN ---
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Tính Loss chung cho mọi model
                loss = criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1)
                    
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                correct_train += (preds == targets).sum().item()
                total_train += targets.size(0)
                
            epoch_train_loss = train_loss / total_train
            epoch_train_acc = correct_train / total_train
            
            # --- VALIDATION ---
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)
                        
                    val_loss += loss.item() * inputs.size(0)
                    correct_val += (preds == targets).sum().item()
                    total_val += targets.size(0)
                    
            epoch_val_loss = val_loss / total_val
            epoch_val_acc = correct_val / total_val
            
            # Lưu lịch sử
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            
            if args.verbose > 0:
                print(f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f}")
                print(f"Val Loss:   {epoch_val_loss:.4f} - Val Acc:   {epoch_val_acc:.4f}")

            # Kiểm tra Early Stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                stop_counter = 0
                best_model_state = model.state_dict().copy()
                if args.verbose > 0:
                    print(f"[*] Val Loss giảm. Đã lưu phiên bản tốt nhất (Best Loss: {best_val_loss:.4f})")
            else:
                stop_counter += 1
                if args.verbose > 0:
                    print(f"[-] Val Loss không cải thiện ({stop_counter}/{stop})")
                    
                if stop_counter >= stop:
                    print(f"[*] Đã kích hoạt Early Stopping tại Epoch {epoch+1}!")
                    break

    except KeyboardInterrupt:
        print("\n[!] NHẬN ĐƯỢC LỆNH NGẮT (Ctrl+C). Đang tiến hành lưu lại phiên bản tốt nhất trước khi thoát...")

    # Trả lại trọng số tốt nhất cho model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("[+] Đã phục hồi trọng số có Validation Loss thấp nhất.")

    plot_curves(history, model_name=args.model.upper())

    # --------------------------------------------------------
    # Lưu trọng số sau khi train xong
    # --------------------------------------------------------
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    weights_dir = project_root / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model == "googlenet22":
        weight_filename = 'googlenet.npy'
    elif args.model == "densenet121":
        weight_filename = 'densenet.npy'
    else:
        weight_filename = f'{args.model}.npy'
        
    weight_path = weights_dir / weight_filename
    torch.save(model.state_dict(), weight_path)
    print(f"Đã lưu thành công trọng số mô hình tại: {weight_path}")


    # clean vram after runnning model
    del model 
    gc.collect()
    torch.cuda.empty_cache()
    print('Cleaned RAM')
    print("Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình phân loại')
    parser.add_argument('--model', type=str, default='googlenet22', choices=['googlenet22', 'densenet121', 'resnet50', 'vgg16', 'vgg19'], help='Chọn mô hình để train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (mặc định: 0.001)')
    parser.add_argument('--epoch', type=int, default=10, help='Số lượng epoch')
    parser.add_argument('--stop', type=int, default=30, help='Số epoch cho Early Stopping (mặc định: 30)')
    parser.add_argument('--worker', type=int, default=2, help='Số lượng worker cho dataloader')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--verbose', type=int, default=2, choices=[0, 1, 2], help='Chế độ hiển thị: 0 (im lặng), 1 (từng epoch), 2 (chi tiết với progress bar)')
    
    args = parser.parse_args()
    train_model(args)
