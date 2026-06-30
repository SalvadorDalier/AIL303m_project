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