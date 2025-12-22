import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_device():
    """
    Get the available device (CPU or GPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path):
    """
    Ensure that a directory exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)
