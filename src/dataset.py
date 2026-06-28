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
