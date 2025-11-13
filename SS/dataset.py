# dataset.py

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

    def __init__(self, csv_path: Path):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)

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

        label = int(row["label_int"])
        label = torch.tensor(label, dtype=torch.long)

        return img, label
