import os
import torch
import random
import numpy as np


SEED = 42
IMG_SIZE = 16
NUM_CORNERS = 4
NUM_CLASSES = 2**NUM_CORNERS - 1  # all non-empty subsets of 4 corners -> 15

TRAIN_SAMPLES = 4000
TEST_SAMPLES = 1000
BATCH_SIZE = 64

LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
FIGS_DIR = os.path.join(ROOT_DIR, "figs")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

EXPERIMENTS = {
    "cnn_10": {"model_type": "cnn_gap", "num_epochs": 10},
    "cnn_50": {"model_type": "cnn_gap", "num_epochs": 50},
    "fcnn_10": {"model_type": "cnn_fc", "num_epochs": 10},
    "transformer_10": {"model_type": "transformer", "num_epochs": 10},
    "hybrid_10": {"model_type": "hybrid", "num_epochs": 10},  # conv stem + transformer
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
