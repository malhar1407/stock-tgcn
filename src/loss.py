"""
loss.py
Directional-penalized MSE loss.
Wrong-direction predictions are penalized by a factor of alpha.
"""

import torch
import torch.nn as nn


class DirectionalMSELoss(nn.Module):
    def __init__(self, alpha: float = 1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2
        wrong_direction = (torch.sign(pred) != torch.sign(target)).float()
        penalty = 1.0 + (self.alpha - 1.0) * wrong_direction  # 1.0 or alpha
        return (mse * penalty).mean()
