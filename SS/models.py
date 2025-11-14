# models.py

import torch.nn as nn
import torch
import torch.nn.functional as F

class Baseline3DCNN(nn.Module):
    """
    3D CNN baseline model for 64x64x64 inputs.

    Architecture:
    - Conv3d(1->8) + ReLU + MaxPool
    - Conv3d(8->16) + ReLU + MaxPool
    - Conv3d(16->32) + ReLU
    - AdaptiveAvgPool3d(1)
    - FC(32 -> num_classes)
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 64 -> 32

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 32 -> 16

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(1),  # -> (B, 32, 1, 1, 1)
        )

        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # (B, 32)
        x = self.classifier(x)   # (B, num_classes)
        return x

class Deep3DCNN(nn.Module):
    """
    Deep 3D CNN model (64x64x64 inputs)
    
    Architecture:
    - 4 convolutional blocks with progressive channel expansion (1->8->16->32->64)
    - Each block: Conv3d + BatchNorm + ReLU + MaxPool3d
    - Global average pooling to reduce spatial dimensions to 1x1x1
    - Dropout + Linear classifier for final prediction
    """
    
    def __init__(self, num_classes=3):
        super().__init__()

        # Feature extraction pipeline
        self.features = nn.Sequential(
            self.block(1, 8),      # Input: (B, 1, 64, 64, 64) -> Output: (B, 8, 32, 32, 32)
            self.block(8, 16),     # (B, 8, 32, 32, 32) -> (B, 16, 16, 16, 16)
            self.block(16, 32),    # (B, 16, 16, 16, 16) -> (B, 32, 8, 8, 8)
            self.block(32, 64),    # (B, 32, 8, 8, 8) -> (B, 64, 4, 4, 4)
            nn.AdaptiveAvgPool3d(1),  # Global pooling: (B, 64, 4, 4, 4) -> (B, 64, 1, 1, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),           # Regularization: randomly zero 30% of activations during training
            nn.Linear(64, num_classes) # Final output: (B, 64) -> (B, num_classes)
        )

    def block(self, in_ch, out_ch):
        """
        Reusable convolutional block.
        
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            
        Returns:
            Sequential module with: Conv3d -> BatchNorm -> ReLU -> MaxPool3d
        """
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),  # Preserve spatial dims, expand channels
            nn.BatchNorm3d(out_ch),                               # Normalize activations for stability
            nn.ReLU(inplace=True),                                # Non-linearity activation
            nn.MaxPool3d(2)                                       # Downsample by factor of 2 (reduces memory)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 1, 64, 64, 64)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.features(x)           # Extract features: (B, 1, 64, 64, 64) -> (B, 64, 1, 1, 1)
        x = torch.flatten(x, 1)        # Flatten spatial dims: (B, 64, 1, 1, 1) -> (B, 64)
        return self.classifier(x)      # Classify: (B, 64) -> (B, num_classes)


# -------------------------------------------
# Basic Residual Block (3D)
# -------------------------------------------
class BasicBlock3D(nn.Module):
    expansion = 1  # used by deeper ResNets

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# -------------------------------------------
# ResNet3D Backbone
# -------------------------------------------
class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super().__init__()

        self.in_channels = 64

        # Initial convolution (Large kernel helps MRI)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 4 stages of residual blocks
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create one ResNet stage."""
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)     # (B,64,32,32,32)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # (B,64,16,16,16)

        x = self.layer1(x)    # → 64 channels
        x = self.layer2(x)    # → 128 channels
        x = self.layer3(x)    # → 256 channels
        x = self.layer4(x)    # → 512 channels

        x = self.avgpool(x)   # → (B,512,1,1,1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# -------------------------------------------
# Factory function for ResNet18 3D
# -------------------------------------------
def ResNet3D18(num_classes=3):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes)