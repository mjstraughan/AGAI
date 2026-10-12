import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.helper_lib.model import get_model

# Load CIFAR10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss, and optimizer
model = get_model("CustomCNN")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Quick training loop (1 epoch for time)
model.train()
for batch_idx, (images, labels) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

# Save the trained weights
torch.save(model.state_dict(), "cifar10_model.pth")
print("Model saved to cifar10_model.pth")