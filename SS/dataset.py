# dataset.py

from pathlib import Path
from typing import Dict
import config
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import math


def random_flip_3d(volume):
    ''' Random flip of the image'''
    if random.random() < 0.5:
        volume = torch.flip(volume, dims=[2])  # flip H
    if random.random() < 0.5:
        volume = torch.flip(volume, dims=[3])  # flip W
    if random.random() < 0.5:
        volume = torch.flip(volume, dims=[1])  # flip D
    return volume


def random_rotate_3d(volume, max_deg=10):
    ''' 3D rotation '''
    # Rotation around Z axis (safest for MRI)
    # angle = random.uniform(-max_deg, max_deg) * math.pi / 180
    # grid = F.affine_grid(
    #     torch.tensor([[
    #         [ math.cos(angle), -math.sin(angle), 0],
    #         [ math.sin(angle),  math.cos(angle), 0],
    #         [ 0,               0,               1]
    #     ]], dtype=torch.float32, device=volume.device),
    #     volume.size(),
    #     align_corners=False
    # )
    # volume = F.grid_sample(volume, grid, padding_mode="border", align_corners=False)
    """3D rotation about Z axis.

    Accepts a tensor of shape (C,D,H,W) or (N,C,D,H,W). Internally adds a batch
    dim for affine_grid/grid_sample and uses a 3x4 affine matrix required for 3D.
    """
    angle = random.uniform(-max_deg, max_deg) * math.pi / 180
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # 3x4 affine matrix (rotation around Z + zero translation)
    theta = torch.tensor([[
        [cos_a, -sin_a, 0.0, 0.0],
        [sin_a,  cos_a, 0.0, 0.0],
        [0.0,    0.0,   1.0, 0.0]
    ]], dtype=torch.float32, device=volume.device)  # shape (1, 3, 4)

    # Ensure input is 5D: (N, C, D, H, W)
    squeezed = False
    if volume.dim() == 4:  # (C, D, H, W)
        volume = volume.unsqueeze(0)  # -> (1, C, D, H, W)
        squeezed = True

    grid = F.affine_grid(theta, volume.size(), align_corners=False)  # expects (N,3,4) for 3D
    volume = F.grid_sample(volume, grid, padding_mode="border", align_corners=False)

    if squeezed:
        volume = volume.squeeze(0)  # back to (C, D, H, W)

    return volume


def random_intensity(volume, scale=0.1):
    '''Small intensity scaling'''
    factor = 1 + random.uniform(-scale, scale)
    return volume * factor


def random_noise(volume, sigma=0.01):
    ''' Add random noise to make it more robust and generalized'''
    noise = torch.randn_like(volume) * sigma
    return volume + noise


def get_test_loader():
    test_set = NPYDataset(config.TEST_CSV, augment=False)
    return DataLoader(test_set, batch_size=1, shuffle=False)


class NPYDataset(Dataset):
    """
    Dataset for 3D MRI volumes stored as .npy files.

    Expects a CSV with at least:
        - 'npy_path': full or relative path to .npy file
        - 'label':   one of ['AD', 'MCI', 'CN']

    Output:
        img:   torch.FloatTensor of shape (1, D, H, W)
        label: torch.LongTensor scalar (0, 1, 2)
    """

    def __init__(self, csv_path: Path, augment=False):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.augment = augment

        # Keep only expected classes
        allowed = ['AD', 'MCI', 'CN']
        self.data = self.data[self.data['label'].isin(allowed)].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(
                f"No valid rows in {self.csv_path}. "
                f"Expected 'label' column with values in {allowed}."
            )

        # Explicit label mapping (stable & interpretable)
        label_order = ['AD', 'MCI', 'CN']  # 0, 1, 2
        self.class_to_idx: Dict[str, int] = {lab: i for i, lab in enumerate(label_order)}
        self.idx_to_class: Dict[int, str] = {i: lab for lab, i in self.class_to_idx.items()}

        # Map string labels to ints
        self.data["label_int"] = self.data["label"].map(self.class_to_idx)

        if self.data["label_int"].isnull().any():
            bad = self.data[self.data["label_int"].isnull()]["label"].unique()
            raise ValueError(f"Unmapped class labels found in CSV: {bad}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        path = Path(row["path"])

        # Load image
        img = np.load(path).astype(np.float32)

        # Defensive cleaning: remove NaNs and infs
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        if img.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {img.shape} at {path}")

        # Add channel dimension → (1, D, H, W)
        img = np.expand_dims(img, axis=0)

        # Convert to torch tensor, enforce contiguous layout (for CUDA stability)
        img = torch.from_numpy(img).float().contiguous()

        # Apply augmentations (only when enabled)
        if getattr(self, "augment", False):
            if config.AUG_FLIP:
                img = random_flip_3d(img)
            img = random_rotate_3d(img, max_deg=config.AUG_ROTATION_DEG)
            img = random_intensity(img, scale=config.AUG_INTENSITY_SCALE)
            img = random_noise(img, sigma=config.AUG_GAUSSIAN_NOISE)


        label = int(row["label_int"])
        label = torch.tensor(label, dtype=torch.long)

        return img, label
