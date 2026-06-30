import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
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
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze feature maps to preserve pre-trained patterns
    for param in model.parameters():
        param.requires_grad = False

    # Swap the final Fully Connected (.fc) layer to output your classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
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