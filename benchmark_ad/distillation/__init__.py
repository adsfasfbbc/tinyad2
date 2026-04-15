from .contrastive_loss import (
    TokenContrastiveLoss,
    AttentionKLLoss,
    CLSSimilarityLoss,
    SpatialContrastiveLoss,
)
from .dispatcher import HeterogeneousDistillationDispatcher
from .route_c_plus.losses import BoundaryAwareSpatialContrastiveLoss, GlobalCosineLoss
from .advanced_paradigm.losses import MGDDecoder, MGDLoss, SPKDLoss

__all__ = [
    "TokenContrastiveLoss",
    "AttentionKLLoss",
    "CLSSimilarityLoss",
    "SpatialContrastiveLoss",
    "HeterogeneousDistillationDispatcher",
    "BoundaryAwareSpatialContrastiveLoss",
    "GlobalCosineLoss",
    "MGDDecoder",
    "MGDLoss",
    "SPKDLoss",
]
