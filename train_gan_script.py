import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.helper_lib.model import get_model
from src.helper_lib.trainer import train_gan

# 1. Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 64
epochs = 10 # This will take a few minutes. Lower to 1 or 2 for a fast, low-quality test.

# 2. Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1] for the Generator's Tanh activation
])

print("Downloading and loading MNIST dataset...")
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

# 3. Initialize Model, Loss, and Optimizers
gan_model = get_model("GAN")
criterion = nn.BCELoss()

# Standard GAN learning rates and betas
lr = 0.0002
optimizer_G = optim.Adam(gan_model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(gan_model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 4. Train and Save
print(f"Starting training on {device} for {epochs} epochs...")
trained_model = train_gan(gan_model, data_loader, criterion, (optimizer_G, optimizer_D), device=device, epochs=epochs)

torch.save(trained_model.state_dict(), "gan_mnist.pth")
print("Training complete! Weights successfully saved to 'gan_mnist.pth'.")