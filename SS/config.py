# config.py
"""
Central place for hyperparameters and paths.
Change things here instead of digging into train.py.
"""

from pathlib import Path

# directory paths
CSV_PATH = Path(r"D:\projects\research\preprocessed_metadata_128.csv")
LOG_DIR = Path("runs")  # for TensorBoard

MODEL_NAME = "resnet3d18"
RUN_NAME = "resnet3d18_v4_128"

#EXPERIMENT_NAME = "3-class-deep-3dcnn-v2"
#CHECKPOINT_DIR = Path("checkpoints")
# MODEL_NAME = "deep3d"
# RUN_NAME = "deep3d_v2"

# csvs after the split
TRAIN_CSV = Path(r"D:\projects\research\train_metadata_128.csv")
VAL_CSV = Path(r"D:\projects\research\val_metadata_128.csv")
TEST_CSV = Path(r"D:\projects\research\test_metadata_128.csv")

# data variables
NUM_CLASSES = 3          # AD / MCI / CN
BATCH_SIZE = 2           # start small and increase when stable
NUM_WORKERS = 0          # increase if you want parallel data loading (e.g. 2, 4)

# training parameters
EPOCHS = 50 
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
RANDOM_SEED = 42
VAL_SPLIT = 0.2

INPUT_SHAPE = (1, 128, 128, 128)
OPTIMIZER = "Adam"
SCHEDULER = "StepLR"

# device config
USE_CUDA = True  # set False to force CPU (for debugging)

# logging and checkpointing
LOG_INTERVAL = 20         # batches between console prints
SAVE_BEST_ONLY = True     # only keep best model (lowest val loss)

# data augmentation
DO_AUGMENT = True

# augmentation params
AUG_ROTATION_DEG = 10         # small rotation
AUG_FLIP = True
AUG_GAUSSIAN_NOISE = 0.01
AUG_INTENSITY_SCALE = 0.1

USE_SCHEDULER = True
SCHEDULER_STEP = 5
SCHEDULER_GAMMA = 0.5

EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10

#tensorboard --logdir runs
