import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCNN(nn.Module):
    """
    1D CNN for Gamma Spectrum Classification.
    Structure based on user provided design:
    - Input: (Batch, 1, 1024)
    - Block 1: Conv1D(16, k=15, s=1) -> ReLU
    - Block 2: Conv1D(32, k=11, s=2) -> ReLU (Downsample A)
    - Block 3: Conv1D(64, k=7, s=2) -> ReLU (Downsample B)
    - Block 4: Conv1D(64, k=5, s=2) -> ReLU (Downsample C)
    - Flatten
    - Dropout(0.5)
    - Dense(64) -> ReLU
    - Output(NumClasses)
    """
    def __init__(self, num_classes=3):
        super(SpectralCNN, self).__init__()
        
        # --- Block 1: Feature Extraction ---
        # Large kernel to cover broad peak shapes
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, stride=1, padding=7) # padding=k//2 to keep size
        
        # --- Block 2: Downsample A ---
        # Stride 2 replaces Pooling
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=2, padding=5)
        
        # --- Block 3: Downsample B ---
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=3)
        
        # --- Block 4: Downsample C ---
        # Extracts deep semantic features (e.g., Compton edges)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2)
        
        # --- Classification Head ---
        self.dropout = nn.Dropout(0.5) # Prevent overfitting
        
        # Calculate flatten size automatically
        # Input 1024 -> /1 -> 1024
        # /2 -> 512
        # /2 -> 256
        # /2 -> 128
        # Final shape: (64 channels, 128 length) -> 64 * 128 = 8192
        self.fc1 = nn.Linear(64 * 128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 1, 1024)
        
        # Block 1
        x = F.relu(self.conv1(x))
        
        # Block 2
        x = F.relu(self.conv2(x))
        
        # Block 3
        x = F.relu(self.conv3(x))
        
        # Block 4
        x = F.relu(self.conv4(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        # Output Layer (return logits, CrossEntropyLoss will handle Softmax)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Test model shape
    model = SpectralCNN(num_classes=3)
    dummy_input = torch.randn(2, 1, 1024)
    output = model(dummy_input)
    print("Model Output Shape:", output.shape)
    print(model)
