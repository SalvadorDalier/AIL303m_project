import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Custom Dataset Wrapper to isolate transformations cleanly
class SplitAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def main():
    # --- 1. Configuration & Setup (Your exact settings) ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 36
    EPOCHS = 10
    LEARNING_RATE = 0.1

    MAIN_FOLDER_NAME = "lemon_dataset" 
    FOLDER_1 = "1000 healthy"
    FOLDER_2 = "1000 unhealthy"

    # Set up the absolute Windows directory path
    DATA_DIR = os.path.join(r"C:\Users\Mai Thanh Binh\OneDrive\Desktop\AIL303\AIL303m_project", MAIN_FOLDER_NAME)
    
    # Explicitly define paths to verify they exist
    PATH_HEALTHY = os.path.join(DATA_DIR, FOLDER_1)
    PATH_UNHEALTHY = os.path.join(DATA_DIR, FOLDER_2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Safety checks for Windows paths
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Main directory not found: {DATA_DIR}")
    if not os.path.exists(PATH_HEALTHY) or not os.path.exists(PATH_UNHEALTHY):
        raise FileNotFoundError(f"Could not find subfolders '{FOLDER_1}' or '{FOLDER_2}' inside {DATA_DIR}")

    # --- 2. Structural Transformations ---
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 3. Auto-Splitting via Sklearn ---
    # Load dataset raw without base transforms
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
    num_classes = len(full_dataset.classes)
    
    print(f"Detected Subfolders/Classes: {full_dataset.classes}")
    print(f"Total Images Found: {len(full_dataset)}")

    # Get dataset indices and target labels for a stratified split
    indices = list(range(len(full_dataset)))
    targets = full_dataset.targets

    # Stratified split ensures an even distribution of healthy/unhealthy images in both sets
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.20, 
        stratify=targets, 
        random_state=42
    )

    # Wrap up subsets with distinct transformations
    train_dataset = SplitAugmentedDataset(Subset(full_dataset, train_idx), transform=train_transforms)
    val_dataset = SplitAugmentedDataset(Subset(full_dataset, val_idx), transform=val_transforms)

    # Data loaders (num_workers set to 0 for flawless Windows execution)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Split completed -> Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # --- 4. Loading ResNet-50 Network ---
    print("Loading Pre-trained ResNet-50 Architecture...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze feature maps to preserve pre-trained patterns
    for param in model.parameters():
        param.requires_grad = False

    # Swap the final Fully Connected (.fc) layer to output our 2 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # --- 5. Loss & Optimization ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # --- 6. Training Loop ---
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