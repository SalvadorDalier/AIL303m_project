

# File: C:\Users\Lenovo\Desktop\AIL303m_project\dataset.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# CAU HINH - DIEU CHINH TAT CA TAI DAY
# ============================================================
BATCH_SIZE   = 32
NUM_WORKERS  = 2
IMG_SIZE     = 224

# Paths
TRAIN_DIR    = r"C:\Users\Lenovo\Desktop\AIL303m_project\data\preprocessed\train"
VALID_DIR    = r"C:\Users\Lenovo\Desktop\AIL303m_project\data\preprocessed\valid"

# ============================================================
# Transforms
# ============================================================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

valid_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================
# DataLoader
# ============================================================
def dataloader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Tra ve train_loader va valid_loader."""
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=VALID_DIR, transform=valid_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader


# ============================================================
# Test nhanh
# ============================================================
if __name__ == "__main__":
    train_loader, valid_loader = dataloader()

    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset

    print(f"Classes: {train_dataset.classes}")
    print(f"Train:   {len(train_dataset)} images")
    print(f"Valid:   {len(valid_dataset)} images")
    print(f"Batch:   {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")

    # Lay 1 batch de kiem tra shape
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels:      {labels[:8]}")


# File: C:\Users\Lenovo\Desktop\AIL303m_project\evaluate.py

import torch
import gc

from model.googlenet22 import googlenet_model
from model.densenet121 import densenet_model
from dataset import dataloader
from utils import compute_precision_recall, print_metrics

def evaluate_all():
    models = [
        {"name": "googlenet22", "weight_path": r"C:\Users\Lenovo\Desktop\AIL303m_project\weights\googlenet.npy"},
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


# File: C:\Users\Lenovo\Desktop\AIL303m_project\README.md

# AIL303m Project — Fruit Quality Classification

Binary image classification of fruit as **healthy** or **unhealthy** using deep learning models with transfer learning.

## Project Structure

```
AIL303m_project/
├── data/
│   ├── raw_image/              # Original unprocessed images
│   ├── preprocessed/           # Preprocessed images
│   ├── train/                  # Training set
│   │   ├── healthy/
│   │   └── unhealthy/
│   └── valid/                  # Validation set
│       ├── healthy/
│       └── unhealthy/
├── src/
│   ├── gradcam.py              # Model benchmarking & Grad-CAM utilities
│   └── model/
│       ├── googlenet22.py      # GoogLeNet (Inception v1) model definition
│       └── densenet121.py      # DenseNet-121 model definition
├── weights/
│   └── googlenet.npy           # Saved model weights
├── refer/
│   └── GoogLeNet-Inception/    # Reference materials for GoogLeNet architecture
├── synthetic-fruit-image-generator/  # CGAN-based synthetic data generation
│   ├── generate_lemons.py      # Script to generate synthetic lemon images
│   ├── cgan_generator_*.h5     # Pre-trained CGAN generator checkpoints
│   ├── comparison.png          # Real vs synthetic comparison
│   └── gradcam.png             # Grad-CAM visualization output
├── requirement.txt             # Python dependencies
└── README.md
```

## Models

### GoogLeNet (Inception v1)

- **File:** `src/model/googlenet22.py`
- Pretrained on ImageNet (`GoogLeNet_Weights.DEFAULT`)
- Final fully-connected layer replaced for binary classification (`num_class=2`)
- Auxiliary logits disabled
- Input size: `(batch, 3, 224, 224)`

### DenseNet-121

- **File:** `src/model/densenet121.py`
- Pretrained on ImageNet (`DenseNet121_Weights.DEFAULT`)
- Classifier layer replaced for binary classification (`num_class=2`)
- Input size: `(batch, 3, 224, 224)`

## Dataset

Binary classification with two classes:

| Class       | Description                |
|-------------|----------------------------|
| `healthy`   | Good-quality fruit images  |
| `unhealthy` | Damaged/defective fruit images |

Data is split into `train/` and `valid/` directories, each containing `healthy/` and `unhealthy/` subfolders.

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate virtual environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### Key Dependencies

| Package        | Purpose                         |
|----------------|---------------------------------|
| `torch`        | PyTorch deep learning framework |
| `torchvision`  | Pretrained models & transforms  |
| `torchinfo`    | Model architecture summaries    |
| `tensorflow`   | CGAN synthetic data generator   |
| `matplotlib`   | Plotting & visualization        |
| `pillow`       | Image loading & processing      |

## Usage

### View model architecture

```bash
# GoogLeNet summary
python src/model/googlenet22.py

# DenseNet-121 summary
python src/model/densenet121.py
```

### Run benchmark / Grad-CAM

```powershell
cd src
python gradcam.py
```

> **Note:** `gradcam.py` uses relative imports (`from model.* import ...`), so it must be run from the `src/` directory.

### Generate synthetic fruit images

```bash
python synthetic-fruit-image-generator/generate_lemons.py
```

## Synthetic Data Generation

The project includes a **Conditional GAN (CGAN)** pipeline for generating synthetic fruit images to augment the training dataset. Pre-trained generator checkpoints are provided at different training epochs:

- `cgan_generator_500.h5`
- `cgan_generator_1000.h5`
- `cgan_generator_1500.h5`

## License

This project is for academic use as part of the **AIL303m** course.


# File: C:\Users\Lenovo\Desktop\AIL303m_project\requirement.txt

a b s l - p y = = 2 . 4 . 0 
 
 a s t u n p a r s e = = 1 . 6 . 3 
 
 a u d i o o p - l t s = = 0 . 2 . 2 
 
 a z u r e - a i - c o n t e n t u n d e r s t a n d i n g = = 1 . 2 . 0 b 2 
 
 a z u r e - a i - d o c u m e n t i n t e l l i g e n c e = = 1 . 0 . 2 
 
 a z u r e - c o r e = = 1 . 4 1 . 0 
 
 a z u r e - i d e n t i t y = = 1 . 2 5 . 3 
 
 b e a u t i f u l s o u p 4 = = 4 . 1 5 . 0 
 
 c e r t i f i = = 2 0 2 6 . 6 . 1 7 
 
 c f f i = = 2 . 0 . 0 
 
 c h a r s e t - n o r m a l i z e r = = 3 . 4 . 7 
 
 c l i c k = = 8 . 4 . 2 
 
 c o b b l e = = 0 . 1 . 4 
 
 c o l o r a m a = = 0 . 4 . 6 
 
 c o l o r e d l o g s = = 1 5 . 0 . 1 
 
 c o n t o u r p y = = 1 . 3 . 3 
 
 c r y p t o g r a p h y = = 4 9 . 0 . 0 
 
 c y c l e r = = 0 . 1 2 . 1 
 
 d e f u s e d x m l = = 0 . 7 . 1 
 
 e t _ x m l f i l e = = 2 . 0 . 0 
 
 f i l e l o c k = = 3 . 2 9 . 4 
 
 f l a t b u f f e r s = = 2 5 . 1 2 . 1 9 
 
 f o n t t o o l s = = 4 . 6 3 . 0 
 
 f s s p e c = = 2 0 2 6 . 6 . 0 
 
 g a s t = = 0 . 7 . 0 
 
 g o o g l e - p a s t a = = 0 . 2 . 0 
 
 g r p c i o = = 1 . 8 1 . 1 
 
 h 5 p y = = 3 . 1 4 . 0 
 
 h u m a n f r i e n d l y = = 1 0 . 0 
 
 i d n a = = 3 . 1 8 
 
 i s o d a t e = = 0 . 7 . 2 
 
 J i n j a 2 = = 3 . 1 . 6 
 
 j o b l i b = = 1 . 5 . 3 
 
 k e r a s = = 3 . 1 4 . 1 
 
 k i w i s o l v e r = = 1 . 5 . 0 
 
 l i b c l a n g = = 1 8 . 1 . 1 
 
 l x m l = = 6 . 1 . 1 
 
 m a g i k a = = 0 . 6 . 3 
 
 m a m m o t h = = 1 . 1 1 . 0 
 
 m a r k d o w n - i t - p y = = 4 . 2 . 0 
 
 m a r k d o w n i f y = = 1 . 2 . 2 
 
 m a r k i t d o w n = = 0 . 1 . 6 
 
 M a r k u p S a f e = = 3 . 0 . 3 
 
 m a t p l o t l i b = = 3 . 1 1 . 0 
 
 m d u r l = = 0 . 1 . 2 
 
 m l _ d t y p e s = = 0 . 5 . 4 
 
 m p m a t h = = 1 . 3 . 0 
 
 m s a l = = 1 . 3 7 . 0 
 
 m s a l - e x t e n s i o n s = = 1 . 3 . 1 
 
 n a m e x = = 0 . 1 . 0 
 
 n a r w h a l s = = 2 . 2 2 . 1 
 
 n e t w o r k x = = 3 . 6 . 1 
 
 n u m p y = = 2 . 5 . 0 
 
 o l e f i l e = = 0 . 4 7 
 
 o n n x r u n t i m e = = 1 . 2 0 . 1 
 
 o p e n c v - p y t h o n - h e a d l e s s = = 4 . 1 3 . 0 . 9 2 
 
 o p e n p y x l = = 3 . 1 . 5 
 
 o p t _ e i n s u m = = 3 . 4 . 0 
 
 o p t r e e = = 0 . 1 9 . 1 
 
 p a c k a g i n g = = 2 6 . 2 
 
 p a n d a s = = 3 . 0 . 3 
 
 p d f m i n e r . s i x = = 2 0 2 6 0 1 0 7 
 
 p d f p l u m b e r = = 0 . 1 1 . 1 0 
 
 p i l l o w = = 1 2 . 2 . 0 
 
 p r o t o b u f = = 7 . 3 5 . 1 
 
 p y c p a r s e r = = 3 . 0 
 
 p y d u b = = 0 . 2 5 . 1 
 
 P y g m e n t s = = 2 . 2 0 . 0 
 
 P y J W T = = 2 . 1 3 . 0 
 
 p y p a r s i n g = = 3 . 3 . 2 
 
 p y p d f i u m 2 = = 5 . 1 0 . 1 
 
 p y r e a d l i n e 3 = = 3 . 5 . 6 
 
 p y t h o n - d a t e u t i l = = 2 . 9 . 0 . p o s t 0 
 
 p y t h o n - d o t e n v = = 1 . 2 . 2 
 
 p y t h o n - p p t x = = 1 . 0 . 2 
 
 r e q u e s t s = = 2 . 3 4 . 2 
 
 r i c h = = 1 5 . 0 . 0 
 
 s c i k i t - l e a r n = = 1 . 9 . 0 
 
 s c i p y = = 1 . 1 8 . 0 
 
 s e t u p t o o l s = = 8 1 . 0 . 0 
 
 s i x = = 1 . 1 7 . 0 
 
 s o u p s i e v e = = 2 . 8 . 4 
 
 S p e e c h R e c o g n i t i o n = = 3 . 1 7 . 0 
 
 s t a n d a r d - a i f c = = 3 . 1 3 . 0 
 
 s t a n d a r d - c h u n k = = 3 . 1 3 . 0 
 
 s y m p y = = 1 . 1 4 . 0 
 
 t e n s o r f l o w = = 2 . 2 1 . 0 
 
 t e r m c o l o r = = 3 . 3 . 0 
 
 t h r e a d p o o l c t l = = 3 . 6 . 0 
 
 t o r c h = = 2 . 1 2 . 1 
 
 t o r c h i n f o = = 1 . 8 . 0 
 
 t o r c h v i s i o n = = 0 . 2 7 . 1 
 
 t q d m = = 4 . 6 8 . 3 
 
 t t a c h = = 0 . 0 . 3 
 
 t y p i n g _ e x t e n s i o n s = = 4 . 1 5 . 0 
 
 t z d a t a = = 2 0 2 6 . 2 
 
 u r l l i b 3 = = 2 . 7 . 0 
 
 w h e e l = = 0 . 4 7 . 0 
 
 w r a p t = = 2 . 2 . 2 
 
 x l r d = = 2 . 0 . 2 
 
 x l s x w r i t e r = = 3 . 2 . 9 
 
 y o u t u b e - t r a n s c r i p t - a p i = = 1 . 0 . 3 
 
 

# File: C:\Users\Lenovo\Desktop\AIL303m_project\train.py

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
    project_root = Path(__file__).resolve().parent
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


# File: C:\Users\Lenovo\Desktop\AIL303m_project\utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CAU HINH - DIEU CHINH TAT CA TAI DAY
# ============================================================
NUM_CLASS    = 2
CLASS_NAMES  = ["healthy", "unhealthy"]
DEVICE       = 'cuda' if torch.cuda.is_available() else print('Cant execute because this model need to be run on GPU ')

# Paths
PROJECT_DIR  = r"C:\Users\Lenovo\Desktop\AIL303m_project"
SAVE_DIR     = PROJECT_DIR + r"\output"

# Plot
FIGURE_SIZE  = (14, 5)
DPI          = 150

# ============================================================
# 1. Precision & Recall
# ============================================================

def compute_precision_recall(model, dataloader, device=DEVICE, num_class=NUM_CLASS):
    """
    Tinh Precision va Recall cho tung class.
    
    Args:
        model: model da train
        dataloader: validation/test dataloader
        device: 'cpu' hoac 'cuda'
        num_class: so luong class (mac dinh: 2)
    
    Returns:
        precision: dict {class_idx: precision_value}
        recall: dict {class_idx: recall_value}
        confusion_matrix: numpy array (num_class x num_class)
    """
    model.eval()
    model.to(device)

    # Confusion matrix: hang = actual, cot = predicted
    confusion = np.zeros((num_class, num_class), dtype=int)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[t][p] += 1

    # Tinh Precision va Recall tu confusion matrix
    precision = {}
    recall = {}

    for c in range(num_class):
        # Precision = TP / (TP + FP)
        tp = confusion[c][c]
        fp = confusion[:, c].sum() - tp      # tong cot c - TP
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall = TP / (TP + FN)
        fn = confusion[c, :].sum() - tp       # tong hang c - TP
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall, confusion


def print_metrics(precision, recall, confusion, class_names=CLASS_NAMES):
    """In Precision, Recall va Confusion Matrix ra terminal."""
    num_class = len(precision)

    print("\n" + "=" * 45)
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10}")
    print("-" * 45)
    for c in range(num_class):
        print(f"  {class_names[c]:<15} {precision[c]:>10.4f} {recall[c]:>10.4f}")
    print("=" * 45)

    print("\nConfusion Matrix:")
    # Header
    header = f"  {'':>12}" + "".join(f"{class_names[c]:>12}" for c in range(num_class))
    print(header)
    for r in range(num_class):
        row = f"  {class_names[r]:>12}" + "".join(f"{confusion[r][c]:>12d}" for c in range(num_class))
        print(row)
    print()


# ============================================================
# 2. Accuracy & Loss Curve
# ============================================================

def plot_curves(history, model_name="model", save_dir=SAVE_DIR):
    """
    Ve bieu do Accuracy va Loss curve.

    Args:
        history: dict voi cac key:
            - 'train_acc': list accuracy training moi epoch
            - 'val_acc': list accuracy validation moi epoch
            - 'train_loss': list loss training moi epoch
            - 'val_loss': list loss validation moi epoch
        model_name: ten model (de dat tieu de va ten file)
        save_dir: thu muc luu file anh. Mac dinh: results/
    """
    epochs = range(1, len(history['train_acc']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    # --- Accuracy Curve ---
    ax1.plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Train Accuracy')
    ax1.plot(epochs, history['val_acc'], 'r-o', markersize=4, label='Val Accuracy')
    ax1.set_title(f'{model_name} - Accuracy Curve', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Loss Curve ---
    ax2.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    ax2.plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val Loss')
    ax2.set_title(f'{model_name} - Loss Curve', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filepath = f"{save_dir}\\{model_name}_curves.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Da luu bieu do tai: {filepath}")

    plt.show()


# ============================================================
# Test nhanh (chay doc lap de kiem tra)
# ============================================================
if __name__ == "__main__":
    # # --- Test plot_curves voi du lieu gia ---
    dummy_history = {
        'train_acc': [55, 65, 72, 78, 83, 87, 90, 92, 93, 94],
        'val_acc':   [50, 60, 68, 73, 77, 80, 82, 84, 85, 85],
        'train_loss': [1.2, 0.9, 0.7, 0.55, 0.42, 0.33, 0.25, 0.19, 0.15, 0.12],
        'val_loss':   [1.3, 1.0, 0.8, 0.65, 0.55, 0.48, 0.44, 0.42, 0.41, 0.40],
    }
    plot_curves(dummy_history, model_name="GoogLeNet_Test")

    # --- Test print_metrics voi du lieu gia ---
    dummy_precision = {0: 0.92, 1: 0.88}
    dummy_recall = {0: 0.90, 1: 0.91}
    dummy_confusion = np.array([[45, 5], [4, 46]])
    print_metrics(dummy_precision, dummy_recall, dummy_confusion)

# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\gradcam.py

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


# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\model\densenet121.py

import torch 
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
num_class = 2
def densenet_model(num_class):
    weight = models.DenseNet121_Weights.DEFAULT # mặc định thấy trọng số tốt nhất
    model = models.densenet121(weights=weight)

    ori_feature = model.classifier.in_features
    model.classifier = nn.Linear(in_features=ori_feature, out_features=num_class)
    return model

if __name__ == "__main__":
    densenet_for_project = densenet_model(num_class)
    INPUT_SIZE = (32, 3, 224, 224)
    print('densenet is running')
    
    summary(
        densenet_for_project,
        input_size=INPUT_SIZE,
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=25,
        row_settings=['var_names']
    )

# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\model\googlenet22.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

def googlenet_model(num_class=2):
    weight = models.GoogLeNet_Weights.DEFAULT
    model = models.googlenet(weight)
    
    model.aux_logits = False
    ori_feature = model.fc.in_features
    model.fc = nn.Linear(in_features=ori_feature, out_features=num_class)
    return model 

if __name__ == "__main__":
    googlenet = googlenet_model(num_class=2)
    INPUT_SIZE = (32,3,224,224)

    print('GoogleNET is running')

    summary(
        googlenet,
        input_size = INPUT_SIZE,
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=25,
        row_settings=['var_names']
    )

# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\module_going\env_setup.py

import sys
import os

def setup_refer_lib():
    """
    Hàm này tự động tìm và thêm thư mục 'refer/pytorch-grad-cam' vào sys.path
    để các file khác có thể import thư viện bên ngoài dễ dàng.
    """
    # Lấy đường dẫn hiện tại (src/module_going)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Lùi về src/
    src_dir = os.path.dirname(current_dir)
    # Lùi về thư mục gốc (AIL303m_project)
    project_root = os.path.dirname(src_dir)
    
    # Trỏ tới thư viện trong refer
    gradcam_lib_path = os.path.join(project_root, 'refer', 'pytorch-grad-cam')
    
    # Add vào system path
    if gradcam_lib_path not in sys.path:
        sys.path.append(gradcam_lib_path)
        
    return project_root


# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\module_going\__init__.py

# File này biến thư mục module_going thành một Python package
