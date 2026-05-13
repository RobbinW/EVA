import random

import numpy as np
import torch


ACTION_CLOSE_LIMITS = torch.tensor(
    [
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.5,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.5,
    ]
)

FLIP_NEGATE_INDICES = torch.tensor([0, 4, 5, 7, 11, 12], dtype=torch.long)


def seed_torch(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flip_action(action: torch.Tensor) -> torch.Tensor:
    flipped = torch.zeros_like(action)
    flipped[..., :7] = action[..., 7:]
    flipped[..., 7:] = action[..., :7]
    flipped[..., FLIP_NEGATE_INDICES] *= -1
    return flipped


def is_close(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    limits = ACTION_CLOSE_LIMITS.to(device=target.device, dtype=target.dtype)
    if target.dim() == 1:
        return torch.all(torch.abs(target - prediction) < limits)
    return torch.all(torch.abs(target - prediction) < limits, dim=1)
