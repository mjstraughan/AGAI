import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# 1. Define the custom CNN architecture (Replace with code from Module4-CNN notebook if required)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()
        # Input channels = 3 (RGB image)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 224x224 image pooled twice becomes 56x56
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Define the Assignment 2 Custom CNN architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adjusted for CIFAR10 32x32 images (becomes 8x8 after two 2x2 pooling layers)
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Implement the routing function
def get_model(model_name):
    if model_name == "CNN":
        # Returns the custom CNN defined above
        model = SimpleCNN()
        
    elif model_name == "CustomCNN":
        # Returns the Assignment 2 CNN for CIFAR10
        model = CustomCNN()
        
    elif model_name == "EnhancedCNN":
        # Returns the pre-trained ResNet18 model imported at the top
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
    elif model_name == "FCNN":
        # Returns a standard Fully Connected Neural Network
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Linear(512, 1000)
        )
        
    else:
        raise ValueError("Invalid model_name. Choose FCNN, CNN, CustomCNN, or EnhancedCNN.")
    
    return model