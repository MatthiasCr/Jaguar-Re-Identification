import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceClassifierForAttribution(nn.Module):
    """Expose cosine class scores for attribution methods."""

    def __init__(self, arcface_model: nn.Module):
        super().__init__()
        self.backbone = arcface_model.backbone
        self.embedding_net = arcface_model.embedding_net
        self.arcface_weight = arcface_model.arcface.weight

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.embedding_net(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.arcface_weight, p=2, dim=1)
        return F.linear(embeddings, weights)


def normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    min_val = heatmap.amin(dim=(-2, -1), keepdim=True)
    max_val = heatmap.amax(dim=(-2, -1), keepdim=True)
    return (heatmap - min_val) / (max_val - min_val + 1e-8)


def attribution_to_heatmap(attribution: torch.Tensor) -> torch.Tensor:
    if attribution.dim() != 4:
        raise ValueError(
            f"Expected attribution with shape [batch, channels, height, width], got {tuple(attribution.shape)}."
        )
    heatmap = attribution.abs().sum(dim=1)
    return normalize_heatmap(heatmap)
