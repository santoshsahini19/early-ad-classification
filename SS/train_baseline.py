# train.py

"""
Core training script

Training details, logs, and checkpoints are all handled here.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
#from torch.utils.tensorboard import SummaryWriter

import config
from dataset import NPYDataset
from models import Baseline3DCNN


def setup_seed(seed: int = 42):
    """
    Make runs as deterministic as possible.
    This is great for debugging; for final training you can relax some of this.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if config.USE_CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    # ---- Setup & environment ----
    setup_seed(config.RANDOM_SEED)

    # Makes CUDA use "safe" conv kernels (good for debugging, esp. 3D convs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: if fully synchronous CUDA is needed for debugging
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    device = get_device()
    print(f"Using device: {device}")

    # Ensure output directories exist
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Dataset & DataLoaders ----
    print("Loading dataset...")
    dataset = NPYDataset(config.CSV_PATH)

    # Stratified train/val split
    labels_int = dataset.data["label_int"].values
    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=labels_int,
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    # ---- Model, Loss, Optimizer ----
    model = Baseline3DCNN(num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # TensorBoard writer
    # writer = SummaryWriter(log_dir=str(config.LOG_DIR))

    best_val_loss = float("inf")

    global_step = 0

    # ---- Training Loop ----
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n==== Epoch {epoch}/{config.EPOCHS} ====")
        # --- Train ---
        model.train()
        train_loss_sum = 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            # Move batch to GPU (or CPU)
            imgs = imgs.to(device).contiguous()
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Logging
            train_loss_sum += loss.item()
            # writer.add_scalar("Loss/train_batch", loss.item(), global_step)

            if batch_idx % config.LOG_INTERVAL == 0:
                print(
                    f"  [Batch {batch_idx:04d}/{len(train_loader):04d}] "
                    f"Loss: {loss.item():.4f}"
                )

            global_step += 1

        avg_train_loss = train_loss_sum / max(1, len(train_loader))
        # writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()

                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)

        # writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        # writer.add_scalar("Accuracy/val_epoch", val_acc, epoch)

        print(
            f"Epoch {epoch}: "
            f"Train Loss = {avg_train_loss:.4f} | "
            f"Val Loss = {avg_val_loss:.4f} | "
            f"Val Acc = {val_acc:.2f}%"
        )

        # --- Checkpointing ---
        ckpt_path = config.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"

        # Always save this epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "config": vars(config),
            },
            ckpt_path,
        )

        # Optionally track "best" model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = config.CHECKPOINT_DIR / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved to {best_path}")

    print("\nTraining complete!")
    # writer.close()


if __name__ == "__main__":
    main()
