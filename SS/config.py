# config.py
"""
Central place for hyperparameters and paths.
Change things here instead of digging into train.py.
"""

from pathlib import Path

# directory paths
CSV_PATH = Path(r"D:\projects\research\preprocessed_metadata_64.csv")
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("runs")  # for TensorBoard

# data variables
NUM_CLASSES = 3          # AD / MCI / CN
BATCH_SIZE = 2           # start small and increase when stable
NUM_WORKERS = 0          # increase if you want parallel data loading (e.g. 2, 4)

# training parameters
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
RANDOM_SEED = 42
VAL_SPLIT = 0.2

# device config
USE_CUDA = True  # set False to force CPU (for debugging)

# logging and checkpointing
LOG_INTERVAL = 20         # batches between console prints
SAVE_BEST_ONLY = True     # only keep best model (lowest val loss)
