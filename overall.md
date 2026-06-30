

# File: C:\Users\Lenovo\Desktop\AIL303m_project\experiment_log.md

# Nhật ký Thử nghiệm Huấn luyện (Experiment Log)

Tài liệu này lưu trữ lịch sử các lần chạy thử nghiệm huấn luyện mô hình, các tham số cấu hình, kết quả đạt được và các ghi chú quan trọng.

## Bảng Tóm tắt Thử nghiệm (Experiment Summary Table)

| STT | Ngày | Tên Mô hình | Batch Size | LR (Learning Rate) | Epochs | Optimizer | Loss (Train/Val) | Acc (Train/Val) | Ghi chú (Note) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-06-28 | GoogLeNet | 32 | 0.001 | 20 | Adam | 0.25 / 0.32 | 92.5% / 89.1% | Lần chạy đầu tiên, mô hình hội tụ tốt. |
| 2 | 2026-06-28 | DenseNet-121 | 16 | 0.0001 | 30 | Adam | 0.18 / 0.24 | 95.2% / 91.8% | Kết quả tốt hơn GoogLeNet, nhưng huấn luyện lâu hơn. |
| 3 | | | | | | | | |đã có add 1 module xóa cache khi chạy |

---
#note
thông tin bài báo:
- Model pruning shows that a Conditional GAN-augmented classification network can retain 81.16% classification accuracy when compressed to 50% of its original size.
- chỉ phân biệt được 'unhealthy synthetic fruit images with defects such as mould and gangrene.'
- Conditional Generative Adversarial Networks ( CGAN ): just create the new synthetic data
- train on 2000 epochs
- we suggest a machine learning pipeline that combines the ideas of fine-tuning, transfer learning, and generative model-based training data 

## Chi tiết từng lần chạy (Detailed Run Logs)

### Thử nghiệm #1: GoogLeNet Baseline
- **Ngày:** 2026-06-28
- **Mô hình:** GoogLeNet (Inception v1)
- **Tham số chi tiết:**
  - Batch Size: 32
  - Learning Rate: 0.001 (không decay)
  - Epochs: 20
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss
  - Image Size: 224x224
- **Kết quả:**
  - Train Loss: 0.25 | Val Loss: 0.32
  - Train Acc: 92.5% | Val Acc: 89.1%
- **Ghi chú / Đánh giá:**
  - Mô hình chạy mượt mà, không gặp lỗi bộ nhớ (OOM).
  - Có dấu hiệu Overfitting nhẹ từ epoch 15 trở đi. Cần xem xét thêm dropout hoặc weight decay ở lần tiếp theo.

---

### Thử nghiệm #2: DenseNet-121
- **Ngày:** 2026-06-28
- **Mô hình:** DenseNet-121
- **Tham số chi tiết:**
  - Batch Size: 16
  - Learning Rate: 0.0001
  - Epochs: 30
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss
- **Kết quả:**
  - Train Loss: 0.18 | Val Loss: 0.24
  - Train Acc: 95.2% | Val Acc: 91.8%
- **Ghi chú / Đánh giá:**
  - Độ chính xác cải thiện rõ rệt so với GoogLeNet.
  - Thời gian train mỗi epoch lâu hơn khoảng 1.5 lần.


# File: C:\Users\Lenovo\Desktop\AIL303m_project\experiment_log_vgg16,19,resnet.md

# Nhật ký Thử nghiệm Huấn luyện (Experiment Log)

Tài liệu này lưu trữ lịch sử các lần chạy thử nghiệm huấn luyện mô hình, các tham số cấu hình, kết quả đạt được và các ghi chú quan trọng.

## Bảng Tóm tắt Thử nghiệm (Experiment Summary Table)

|STT|Ngày|Tên Mô hình|Batch Size|LR (Learning Rate)|Epochs|Optimizer|Loss (Train/Val)|Acc (Train/Val)|Ghi chú (Note)|
|-|-|-|-|-|-|-|-|-|-|
|1|2026-06-28|VGG16|36|0.001|5|Adam|0.26652 / 0.18486|92.8% / 93.2%|chạy lâu, độ chính xác cao nhất|
|2|2026-06-28|VGG19|36|0.001|5|Adam|0.24 / 0.187|80.64% / 92.8%|lâu hơn, chính xác thấp hơn vgg16|
|3|2026-06-29|Restnet50|36|0,001|5|Adam|0,40/0,40|82,13% / 83,1%|chạy nhanh hơn vgg16\&19, độ chính xác trên val thấp nhất|

\---

## Chi tiết từng lần chạy (Detailed Run Logs)

### Thử nghiệm #1: VGG16

* **Ngày:** 2026-06-28
* **Tham số chi tiết:**

  * Batch Size: 36
  * Learning Rate: 0.001
  * Epochs: 5
  * Optimizer: Adam
  * Loss Function: CrossEntropyLoss
  * Image Size: 224x224
* **Kết quả:**

  * Train Loss: 0.26 | Val Loss: 0.18
  * Train Acc: 92.8% | Val Acc: 93.2%
* **Ghi chú / Đánh giá:**

  * Mô hình chạy lâu, 1 batch mất gần 5 phút
  * độ chính xác cao nhất (có thể do data đơn giản)

### Thử nghiệm #2: VGG19

* **Ngày:** 2026-06-28
* **Tham số chi tiết:**

  * Batch Size: 5
  * Learning Rate: 0.001
  * Epochs: 5
  * Optimizer: Adam
  * Loss Function: CrossEntropyLoss
* **Kết quả:**

  * Train Loss: 0.24 | Val Loss: 0.18
  * Train Acc: 80.64% | Val Acc: 92.8%
* **Ghi chú / Đánh giá:**

  * Độ chính xác trên val không cải thiện nhiều, train acc thấp hơn so với vgg16 rất nhiều (có thể do epoch quá thấp)
  * Thời gian chạy lâu hơn vgg19

### Thử nghiệm #3: Restnet50

* **Ngày:** 2026-06-29
* **Tham số chi tiết:**

  * Batch Size: 5
  * Learning Rate: 0.001
  * Epochs: 5
  * Optimizer: Adam
  * Loss Function: CrossEntropyLoss
* **Kết quả:**

  * Train Loss: 0.4 | Val Loss: 0.4
  * Train Acc: 82.13% | Val Acc: 83.1%
* **Ghi chú / Đánh giá:**

  * Mô hình chạy nhanh hơn cả vgg16\&19
  * Train và val acc lại thấp hơn cả 2 vgg (có thể do thiếu data, chạy ít epoch)



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

### Supported Architectures:
- **GoogLeNet (Inception v1):** `src/model/googlenet22.py`
- **DenseNet-121:** `src/model/densenet121.py`
- **ResNet-50:** `src/model/resnet50.py`
- **VGG-16:** `src/model/vgg16.py`
- **VGG-19:** `src/model/vgg19.py`

All models are pretrained on ImageNet and have their final fully-connected layers adapted for binary classification (`num_class=2`). Expected input tensor size is `(batch, 3, 224, 224)`.

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

### Train a model

You can train a model using `train.py` inside the `src/` directory.

```powershell
cd src
# Train GoogLeNet (Inception v1) for 10 epochs
python train.py --model googlenet22 --epoch 10 --batch 32

# Train DenseNet-121 for 15 epochs with a progress bar
python train.py --model densenet121 --epoch 15 --batch 16 --verbose 2
```

Arguments:
- `--model`: Choose model (`googlenet22`, `densenet121`, `resnet50`, `vgg16`, `vgg19`). Default is `googlenet22`.
- `--lr`: Learning rate for Adam optimizer. Default is `0.001` (Use `0.0001` for full fine-tuning).
- `--epoch`: Maximum number of training epochs. Default is `10`.
- `--stop`: Number of epochs for Early Stopping patience. Default is `30`.
- `--batch`: Batch size. Default is `32`.
- `--worker`: Number of worker threads for dataloader. Default is `2`.
- `--verbose`: Display mode: `0` (silent), `1` (epoch summaries), `2` (detailed progress bar). Default is `2`.

### Train on Cloud GPU (Serverless with Modal)

For massive workloads (e.g., 2000 epochs) without a local GPU, train on cloud GPUs (NVIDIA A10G) using [Modal](https://modal.com). The script automatically syncs your data, trains remotely, and downloads the `.npy` weights and `.png` curves back locally!

```bash
# 1. Login to Modal (Only needed once)
modal setup

# 2. Start Cloud Training
modal run src/train_modal.py --model resnet50 --epoch 2000 --lr 0.0001 --stop 30
```

### Evaluate models

To evaluate the trained models on the validation dataset:

```powershell
cd src
python evaluate.py
```

### Run Explainable AI (Grad-CAM)

```powershell
cd src
python explain.py --model vgg19 --img ../data/preprocessed/valid/healthy/0001_0001.jpg
```

> **Note:** Scripts inside `src/` (`train.py`, `evaluate.py`, `gradcam.py`) use relative imports, so they must be run from the `src/` directory.

### Generate synthetic fruit images

```bash
python synthetic-fruit-image-generator/generate_lemons.py
```

## Synthetic Data Generation

The project includes a **Conditional GAN (CGAN)** pipeline for generating synthetic fruit images to augment the training dataset. Pre-trained generator checkpoints are provided at different training epochs:

- `cgan_generator_500.h5`
- `cgan_generator_1000.h5`
- `cgan_generator_1500.h5`

## Tracking Experiments

We keep a diary/log of all training runs, model configurations, hyperparameters, and results to monitor progress and avoid redundant experiments.

Please document every training attempt in:
- **File:** [`experiment_log.md`](experiment_log.md)

### What to record:
1. **Date & Time:** When the run was executed.
2. **Model Details:** Model name, configuration modifications.
3. **Hyperparameters:** Batch size, learning rate (LR), total epochs, optimizer type.
4. **Metrics:** Best Train/Val Loss, best Train/Val Accuracy.
5. **Notes (Crucial):** Observations (e.g., overfitting behavior, training speed, anomalies, potential next steps).

---

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

# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\dataset.py

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
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR    = str(PROJECT_ROOT / "data" / "preprocessed" / "train")
VALID_DIR    = str(PROJECT_ROOT / "data" / "preprocessed" / "valid")

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


# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\evaluate.py

import torch
import gc

from model.googlenet22 import googlenet_model
from model.densenet121 import densenet_model
from model.resnet50 import resnet50_model
from model.vgg16 import vgg16_model
from model.vgg19 import vgg19_model
from dataset import dataloader
from utils import compute_precision_recall, print_metrics

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def evaluate_all():
    models = [
        {"name": "googlenet22", "weight_path": str(PROJECT_ROOT / "weights" / "googlenet.npy")},
        {"name": "densenet121", "weight_path": str(PROJECT_ROOT / "weights" / "densenet.npy")},
        {"name": "resnet50", "weight_path": str(PROJECT_ROOT / "weights" / "resnet50.npy")},
        {"name": "vgg16", "weight_path": str(PROJECT_ROOT / "weights" / "vgg16.npy")},
        {"name": "vgg19", "weight_path": str(PROJECT_ROOT / "weights" / "vgg19.npy")}
    ]
    
    for m in models:
        evaluate_model(m["name"], m["weight_path"])

def get_model(name, num_class=2, weight_path=None):
    if name == "googlenet22": 
        model = googlenet_model(num_class)
    elif name == "densenet121": 
        model = densenet_model(num_class)
    elif name == "resnet50": 
        model = resnet50_model(num_class)
    elif name == "vgg16": 
        model = vgg16_model(num_class)
    elif name == "vgg19": 
        model = vgg19_model(num_class)
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


# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\explain.py

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


# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\train.py

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
            
        # Kiểm tra Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            stop_counter = 0
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
            if args.verbose in [1, 2]:
                print(f"[Best Model] Đã ghi nhận mô hình tốt nhất với Val Loss: {best_val_loss:.4f}")
        else:
            stop_counter += 1
            if args.verbose in [1, 2]:
                print(f"[Early Stopping] Số epoch liên tiếp không cải thiện: {stop_counter}/{stop}")
                
        if args.verbose in [1, 2]:
            print('-' * 60)
            
        if stop_counter >= stop:
            print(f"\n[!] Dừng sớm kích hoạt! Validation loss không cải thiện sau {stop} epochs liên tục.")
            print(f"[!] Dừng quá trình train tại Epoch {epoch}.")
            break

    # Phục hồi trọng số tốt nhất trước khi vẽ đồ thị và lưu file
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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


# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\train_modal.py

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
        "pillow"
    )
)

# 3. Định vị thư mục dự án cục bộ để tải lên Cloud
local_project_dir = Path(__file__).resolve().parent.parent

# Cấu hình Mount để tải toàn bộ thư mục dự án lên Cloud, loại bỏ các thư mục rác/nặng
project_mount = modal.Mount.from_local_dir(
    local_path=str(local_project_dir),
    remote_path="/root/project",
    condition=lambda p: not any(x in p for x in [".venv", ".git", "__pycache__", "weights", "output"])
)

# 4. Hàm huấn luyện chạy trên GPU của Modal Cloud
@app.function(
    image=image,
    mounts=[project_mount],
    gpu="A10G",          # Sử dụng GPU NVIDIA A10G (24GB VRAM) hiệu năng cao
    timeout=7200         # Giới hạn thời gian chạy tối đa là 2 tiếng (7200s)
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

    # Tạo class giả lập Argument Parser để truyền vào train_model
    class Args:
        model = model_name
        lr = lr
        epoch = epoch
        stop = stop
        worker = 2
        batch = batch
        # Chọn verbose=1 để in kết quả gọn gàng sau mỗi epoch (tránh spam log ở console Cloud)
        verbose = 1

    # Thực thi quá trình train
    train_model(Args())

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


# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CAU HINH - DIEU CHINH TAT CA TAI DAY
# ============================================================
NUM_CLASS    = 2
CLASS_NAMES  = ["healthy", "unhealthy"]
DEVICE       = 'cuda' if torch.cuda.is_available() else print('Cant execute because this model need to be run on GPU ')

from pathlib import Path
# Paths
PROJECT_DIR  = str(Path(__file__).resolve().parent.parent)
SAVE_DIR     = str(Path(PROJECT_DIR) / "output")

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

# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\model\densenet121.py

import torch 
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
num_class = 2
def densenet_model(num_class):
    weight = models.DenseNet121_Weights.DEFAULT # mặc định thấy trọng số tốt nhất
    model = models.densenet121(weights=weight)

    # Đóng băng các lớp convolution (Feature Extraction)
    for param in model.parameters():
        param.requires_grad = False

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
    
    # Đóng băng các lớp convolution (Feature Extraction)
    for param in model.parameters():
        param.requires_grad = False

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

# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\model\resnet50.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def resnet50_model(num_class=2):
    """
    Khởi tạo mô hình ResNet-50 cho pipeline dự án.
    Giữ nguyên logic của partner (freeze feature maps).
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze feature maps to preserve pre-trained patterns
    for param in model.parameters():
        param.requires_grad = False

    # Swap the final Fully Connected (.fc) layer to output your classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_class)
    return model


def main():
    """
    Pipeline huấn luyện gốc của partner.
    """
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 36
    EPOCHS = 5
    LEARNING_RATE = 0.001  

    train_dir = r"C:\Users\Mai Thanh Binh\OneDrive\Desktop\AIL303\AIL303m_project\lemon_dataset\train"
    val_dir = r"C:\Users\Mai Thanh Binh\OneDrive\Desktop\AIL303\AIL303m_project\lemon_dataset\valid"

    data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms)
    
    num_classes = len(train_dataset.classes)
    print(f"Detected Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Loading Pre-trained ResNet-50 Architecture...")
    model = resnet50_model(num_class=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("\nStarting Training Sequence...")
    for epoch in range(EPOCHS):
        # --- Train Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = (correct_train / total_train) * 100

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_val / total_val) * 100

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

    print("\nTraining complete! Saving checkpoint to 'resnet50_lemon_model.pth'...")
    torch.save(model.state_dict(), "resnet50_lemon_model.pth")

if __name__ == "__main__":
    main()

# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\model\vgg16.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def vgg16_model(num_class=2):
    """
    Khởi tạo mô hình VGG16 cho pipeline dự án.
    Giữ nguyên logic của partner (freeze features, thêm Dropout).
    """
    vgg_weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=vgg_weights)

    # Freeze the convolutional base parameters
    for param in model.features.parameters():
        param.requires_grad = False

    num_features = model.classifier[0].in_features 
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_class) 
    )
    return model


def main():
    """
    Pipeline huấn luyện gốc của partner.
    """
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 36
    EPOCHS = 5
    LEARNING_RATE = 0.001

    train_dir = r"C:\Users\Mai Thanh Binh\OneDrive\Desktop\AIL303\AIL303m_project\lemon_dataset\train"
    val_dir = r"C:\Users\Mai Thanh Binh\OneDrive\Desktop\AIL303\AIL303m_project\lemon_dataset\valid"

    data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    num_classes = len(train_dataset.classes)
    model = vgg16_model(num_class=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE) 

    # ---Training Loop ---
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predictions = outputs.max(1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predictions = outputs.max(1)
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} |"
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    # ---Save Model Weights ---
    torch.save(model.state_dict(), 'custom_vgg16_model.pth')
    print("\nTraining complete! Weights saved successfully as 'custom_vgg16_model.pth'")

if __name__ == "__main__":
    main()

# File: C:\Users\Lenovo\Desktop\AIL303m_project\src\model\vgg19.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def vgg19_model(num_class=2):
    """
    Khởi tạo mô hình VGG19 cho pipeline dự án.
    Giữ nguyên logic của partner (freeze features, thêm Dropout).
    """
    vgg_weights = models.VGG19_Weights.DEFAULT
    model = models.vgg19(weights=vgg_weights)

    # Freeze the convolutional base parameters of VGG19
    for param in model.features.parameters():
        param.requires_grad = False

    num_features = model.classifier[0].in_features 
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_class) 
    )
    return model


def main():
    """
    Pipeline huấn luyện gốc của partner.
    """
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 36
    EPOCHS = 5
    LEARNING_RATE = 0.001

    train_dir = r"C:\Users\Mai Thanh Binh\OneDrive\Desktop\AIL303\AIL303m_project\lemon_dataset\train"
    val_dir = r"C:\Users\Mai Thanh Binh\OneDrive\Desktop\AIL303\AIL303m_project\lemon_dataset\valid"

    data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    num_classes = len(train_dataset.classes)
    model = vgg19_model(num_class=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE) 

    print("\nStarting VGG19 Training...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predictions = outputs.max(1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predictions = outputs.max(1)
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} -> "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")

    # Save updated weights
    torch.save(model.state_dict(), 'custom_vgg19_model.pth')
    print("\nTraining complete! Weights saved successfully as 'custom_vgg19_model.pth'")

if __name__ == "__main__":
    main()

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
