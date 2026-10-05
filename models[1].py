"""
Deep Metric Learning Architecture for Color-Invariant Saree Recognition.
Integrates standard vision backbones (ConvNeXt, EfficientNet, ResNet) with
a metric projection head for unit-hypersphere embeddings.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

import torchvision.models as tv_models


class SareeMetricNet(nn.Module):
    """
    Embedding Network for Saree Surface Motif Recognition.
    
    Architecture:
      Backbone (ConvNeXt-Tiny / EfficientNet-B0 / ResNet-50)
        -> Global Average Pooling
        -> Batch Normalization & Dropout
        -> Dense Projection (512-dim)
        -> L2 Normalization (||e||_2 = 1)
    """
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        
        # Load backbone
        if HAS_TIMM:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,  # removes default classifier, gives pooled feature
            )
            in_features = self.backbone.num_features
        else:
            # Fallback to torchvision
            if "resnet50" in backbone_name:
                weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
                base = tv_models.resnet50(weights=weights)
                in_features = base.fc.in_features
                base.fc = nn.Identity()
                self.backbone = base
            elif "convnext" in backbone_name:
                weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                base = tv_models.convnext_tiny(weights=weights)
                in_features = base.classifier[2].in_features
                base.classifier = nn.Identity()
                self.backbone = base
            else:
                weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                base = tv_models.efficientnet_b0(weights=weights)
                in_features = base.classifier[1].in_features
                base.classifier = nn.Identity()
                self.backbone = base

        # Metric projection head
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass extracting L2-normalized metric embeddings.
        Args:
            x: [Batch, 3, H, W] input image batch
            normalize: whether to apply L2 normalization to unit hypersphere
        Returns:
            [Batch, embedding_dim] metric embeddings
        """
        features = self.backbone(x)
        # Flatten if needed
        if features.dim() > 2:
            features = torch.flatten(features, 1)
            
        embeddings = self.neck(features)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings


def build_model(
    backbone_name: str = "convnext_tiny",
    embedding_dim: int = 512,
    pretrained: bool = True,
    dropout: float = 0.2,
    device: Optional[torch.device] = None,
) -> SareeMetricNet:
    """Convenience factory function."""
    model = SareeMetricNet(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model
