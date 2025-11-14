# models.py

import torch.nn as nn
import torch


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
