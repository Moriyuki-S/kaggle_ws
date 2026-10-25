import os
import random
import time
from contextlib import contextmanager

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    print(f"[{name}] {time.perf_counter() - start:.1f}s")
