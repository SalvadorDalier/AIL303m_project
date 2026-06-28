import torch
import gc
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model.googlenet22 import googlenet_model
from model.densenet121 import densenet_model
from dataset import dataloader
from utils import plot_curves

# implement far more 3 models
def get_model(name, num_class=2):
    if name == "googlenet22": return googlenet_model(num_class)
    if name == "densenet121": return densenet_model(num_class)

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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epoch + 1):
        if args.verbose == 2:
            print(f"Epoch {epoch}/{args.epoch}")
            
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epoch} [Train]") if args.verbose == 2 else train_loader
        
        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if args.verbose == 2:
                train_iterator.set_postfix(loss=loss.item(), acc=100.*correct_train/total_train)
                
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_iterator = tqdm(valid_loader, desc=f"Epoch {epoch}/{args.epoch} [Valid]") if args.verbose == 2 else valid_loader
        
        with torch.no_grad():
            for inputs, labels in val_iterator:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(valid_loader.dataset)
        epoch_val_acc = 100. * correct_val / total_val
        
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        
        if args.verbose in [1, 2]:
            print(f"Epoch {epoch}/{args.epoch} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
            print('-' * 60)

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
    else:
        weight_filename = 'densenet.npy'
        
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
    parser.add_argument('--model', type=str, default='googlenet22', choices=['googlenet22', 'densenet121'], help='Chọn mô hình để train')
    parser.add_argument('--epoch', type=int, default=10, help='Số lượng epoch')
    parser.add_argument('--worker', type=int, default=2, help='Số lượng worker cho dataloader')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--verbose', type=int, default=2, choices=[0, 1, 2], help='Chế độ hiển thị: 0 (im lặng), 1 (từng epoch), 2 (chi tiết với progress bar)')
    
    args = parser.parse_args()
    train_model(args)
