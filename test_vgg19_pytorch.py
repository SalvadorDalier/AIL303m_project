import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
BATCH_SIZE = 36
EPOCHS = 10
LEARNING_RATE = 0.1

MAIN_FOLDER_NAME = "lemon_dataset" 
FOLDER_1 = "1000 healthy"
FOLDER_2 = "1000 unhealthy"

DATA_DIR = os.path.join(r"C:\Users\Mai Thanh Binh\OneDrive\Desktop\AIL303\AIL303m_project", MAIN_FOLDER_NAME)
MY_CLASSES = [FOLDER_1, FOLDER_2]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Scanning directory and filtering folders...")
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)


target_class_idx = [full_dataset.class_to_idx[cls] for cls in MY_CLASSES if cls in full_dataset.class_to_idx]
filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in target_class_idx]


for i in filtered_indices:
    original_label = full_dataset.samples[i][1]
    full_dataset.samples[i] = (full_dataset.samples[i][0], target_class_idx.index(original_label))


filtered_dataset = Subset(full_dataset, filtered_indices)

train_idx, val_idx = train_test_split(
    range(len(filtered_dataset)),
    test_size=0.2,
    random_state=123,
    stratify=[full_dataset.samples[i][1] for i in filtered_indices]
)

train_dataset = Subset(filtered_dataset, train_idx)
val_dataset = Subset(filtered_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")

vgg_weights = models.VGG19_Weights.DEFAULT
model = models.vgg19(weights=vgg_weights)

# Freeze the convolutional base parameters of VGG19
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classification "head" for Binary Classification (Output size = 1)
num_features = model.classifier[0].in_features 
model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1) 
)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE) 

print("\nStarting VGG19 Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = correct_val / total_val
    
    print(f"Epoch {epoch+1}/{EPOCHS} -> "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")


torch.save(model.state_dict(), 'custom_vgg19_model.pth')
print("\nTraining complete! Weights saved successfully as 'custom_vgg19_model.pth'")