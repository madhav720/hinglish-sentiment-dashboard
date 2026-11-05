from src.losses.focal_loss import (
    numpy_cross_entropy,
    numpy_focal_loss,
    numpy_softmax,
)

try:
    from src.losses.focal_loss import FocalLoss, ClassWeightedCrossEntropyLoss
except ImportError:
    pass
