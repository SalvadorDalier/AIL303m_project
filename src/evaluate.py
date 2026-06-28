import torch
import gc

from model.googlenet22 import googlenet_model
from model.densenet121 import densenet_model
from dataset import dataloader
from utils import compute_precision_recall, print_metrics

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def evaluate_all():
    models = [
        {"name": "googlenet22", "weight_path": str(PROJECT_ROOT / "weights" / "googlenet.npy")},
        {"name": "densenet121", "weight_path": None}
    ]
    
    for m in models:
        evaluate_model(m["name"], m["weight_path"])

def get_model(name, num_class=2, weight_path=None):
    if name == "googlenet22": 
        model = googlenet_model(num_class)
    elif name == "densenet121": 
        model = densenet_model(num_class)
    else: 
        raise ValueError(f'không có model: {name}')
        
    if weight_path:
        import os
        if os.path.exists(weight_path):
            print(f"[*] Loading weights from: {weight_path}")
            try:
                # Load PyTorch checkpoint
                state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
                if 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'])
                elif 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
                print("[+] Successfully loaded PyTorch weights.")
            except Exception as e:
                print(f"[-] Could not load as PyTorch weights: {e}")
                # Hỗ trợ thông báo nếu file là numpy (.npy)
                if weight_path.endswith('.npy'):
                    print(f"[-] File là định dạng Numpy (.npy). Cần hàm mapping trọng số thủ công cho PyTorch model.")
        else:
            print(f"[-] Trọng số tại {weight_path} không tồn tại. Đang dùng pretrained mặc định của ImageNet.")
            
    return model

def evaluate_model(model_name, weight_path=None):
    print(f"\n" + "="*60)
    print(f"Evaluating {model_name.upper()}")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Khoi tao model va load weights (neu co)
    model = get_model(model_name, num_class=2, weight_path=weight_path)
    model = model.to(device)
    
    # Load data
    _, valid_loader = dataloader()
    
    print("Computing metrics on validation set...")
    precision, recall, confusion = compute_precision_recall(
        model, 
        valid_loader, 
        device=device, 
        num_class=2
    )
    
    # In ket qua
    print_metrics(precision, recall, confusion)
    
    # Don dep RAM
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Cleaned RAM\n')



if __name__ == "__main__":
    evaluate_all()
