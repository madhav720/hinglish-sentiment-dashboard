"""
Loss Functions Implementation
Implements:
1. Standard Multi-Class Cross-Entropy Loss
2. Class-Weighted Multi-Class Cross-Entropy Loss (weights from training split only)
3. Multi-Class Focal Loss (Lin et al., 2017) with configurable gamma and numerical stability.
Includes both PyTorch nn.Module and NumPy reference implementations for cross-validation and testing.
"""

import math
from typing import Optional, List, Dict, Union

# Attempt PyTorch import; provide pure NumPy fallback for unit tests and smoke tests
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


if TORCH_AVAILABLE:
    class FocalLoss(nn.Module):
        """
        Multi-Class Focal Loss (Lin et al., 2017).
        FL = - alpha_t * (1 - p_t)^gamma * log(p_t)
        Where p_t is the softmax probability of the ground-truth class.
        When gamma = 0 and alpha = None, this is mathematically identical to Standard Cross-Entropy.
        """
        def __init__(
            self,
            gamma: float = 2.0,
            alpha: Optional[Union[torch.Tensor, List[float]]] = None,
            reduction: str = "mean",
        ):
            super().__init__()
            self.gamma = float(gamma)
            self.reduction = reduction

            if alpha is not None:
                if isinstance(alpha, list):
                    alpha = torch.tensor(alpha, dtype=torch.float32)
                self.register_buffer("alpha", alpha)
            else:
                self.alpha = None

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Args:
                logits: [batch_size, num_classes] raw unnormalized prediction scores
                targets: [batch_size] ground-truth class indices (0 to num_classes-1)
            """
            num_classes = logits.size(-1)
            # Numerically stable log-softmax
            log_p = F.log_softmax(logits, dim=-1)
            # Gather log probability of the true target class
            log_pt = log_p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
            # Probability pt = exp(log_pt), clamped for numerical safety
            pt = log_pt.exp().clamp(min=1e-8, max=1.0)

            # Modulating factor (1 - pt)^gamma
            focal_weight = (1.0 - pt) ** self.gamma
            loss = -focal_weight * log_pt

            # Optional alpha class weighting
            if self.alpha is not None:
                alpha_t = self.alpha.gather(dim=0, index=targets)
                loss = alpha_t * loss

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise ValueError(f"Unsupported reduction: {self.reduction}")


    class ClassWeightedCrossEntropyLoss(nn.Module):
        """
        Class-Weighted Cross-Entropy Loss.
        L = - w_{y_i} * log(p_{y_i})
        Weights must be derived strictly from the training split.
        """
        def __init__(
            self,
            weights: Union[torch.Tensor, List[float]],
            reduction: str = "mean",
        ):
            super().__init__()
            if isinstance(weights, list):
                weights = torch.tensor(weights, dtype=torch.float32)
            self.register_buffer("weights", weights)
            self.reduction = reduction

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(logits, targets, weight=self.weights, reduction=self.reduction)


# NumPy Reference Implementation for Verification & Smoke Testing
def numpy_softmax(logits: List[List[float]]) -> List[List[float]]:
    """Compute numerically stable softmax over rows."""
    probs = []
    for row in logits:
        max_val = max(row)
        exp_vals = [math.exp(v - max_val) for v in row]
        sum_exp = sum(exp_vals)
        probs.append([v / sum_exp for v in exp_vals])
    return probs


def numpy_cross_entropy(
    logits: List[List[float]],
    targets: List[int],
    weights: Optional[List[float]] = None,
) -> float:
    """Compute Standard or Weighted Cross-Entropy in pure Python/NumPy."""
    probs = numpy_softmax(logits)
    losses = []
    for p_row, y in zip(probs, targets):
        pt = max(p_row[y], 1e-12)
        loss = -math.log(pt)
        if weights is not None:
            loss = loss * weights[y]
        losses.append(loss)
    return sum(losses) / len(losses) if losses else 0.0


def numpy_focal_loss(
    logits: List[List[float]],
    targets: List[int],
    gamma: float = 2.0,
    alpha: Optional[List[float]] = None,
) -> float:
    """Compute Multi-Class Focal Loss in pure Python/NumPy."""
    probs = numpy_softmax(logits)
    losses = []
    for p_row, y in zip(probs, targets):
        pt = max(min(p_row[y], 1.0), 1e-12)
        log_pt = math.log(pt)
        modulating_factor = (1.0 - pt) ** gamma
        loss = -modulating_factor * log_pt
        if alpha is not None:
            loss = loss * alpha[y]
        losses.append(loss)
    return sum(losses) / len(losses) if losses else 0.0
