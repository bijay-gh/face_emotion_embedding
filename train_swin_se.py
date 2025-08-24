from model import *
import torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
from torch import nn, optim
import yaml
import os
import time 

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  


# Transform (convert grayscale to 3-channel for Swin)
transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load datasets from folders
train_data = datasets.ImageFolder(root=config['dateset']['FER2013']['train'], transform=transform)
test_data = datasets.ImageFolder(root=config['dateset']['FER2013']['test'], transform=transform)

# Split training data: 90% train, 10% validation
train_len = int(0.9 * len(train_data))
val_len = len(train_data) - train_len
train_dataset, val_dataset = random_split(train_data, [train_len, val_len])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class_names = train_data.classes  

# Initialize model, loss function, optimizer
model = SwinWithSE(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
num_epochs = config['epochs']

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
best_val_acc = 0.0

for epoch in range(num_epochs):
    start_time = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    # Save best model
    if epoch > 10 and val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        print(f"Best model saved at epoch {epoch+1} with val acc: {val_acc:.4f}")

    end_time = time.time()
    print(f"Epoch {epoch+1}/{num_epochs} - Time: {end_time-start_time:.2f}s")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")


test_acc, test_f1 = evaluate_test(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")