"""
src/utils/seed.py
Centralized reproducibility. Call set_seed() before anything else.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic ops — slight performance cost, worth it for research
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False