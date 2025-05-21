import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import swin_t, Swin_T_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class SwinWithSE(nn.Module):
    def __init__(self, num_classes):
        super(SwinWithSE, self).__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()  # Remove original classification head

        self.se = SEBlock(in_features)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.se(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x
    

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc