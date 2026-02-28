import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EmbeddingProjection(nn.Module):
    """
    Projects backbone embeddings to a lower-dimensional space.
    Architecture: input_dim -> hidden_dim -> output_dim
    """

    def __init__(self, input_dim=1536, hidden_dim=512, output_dim=256, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)


class ArcFaceLayer(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) layer.

    The loss is computed as:
        L = -log(exp(s * cos(theta_y + m)) / (exp(s * cos(theta_y + m)) + sum(exp(s * cos(theta_j)))))

    where:
        - theta_y is the angle between embedding and ground truth class center
        - m is the angular margin (default 0.5 radians, about 28.6 degrees)
        - s is the feature scale (default 64)
    """

    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight_norm)
        cosine = cosine.clamp(-1.0, 1.0)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        return output


def build_backbone(backbone_name: str, pretrained: bool = True):
    return timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)


class ArcFaceModel(nn.Module):
    """Backbone + projection + ArcFace."""

    def __init__(
        self,
        backbone_name,
        num_classes,
        embedding_dim=256,
        hidden_dim=512,
        margin=0.5,
        scale=64.0,
        dropout=0.3,
        pretrained=True,
        backbone_out_dim=None,
        freeze_backbone=True,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if backbone_out_dim is None:
            backbone_out_dim = getattr(self.backbone, "num_features", None)
        if backbone_out_dim is None:
            raise ValueError("Backbone output dimension not found; pass backbone_out_dim")

        self.embedding_net = EmbeddingProjection(
            input_dim=backbone_out_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        self.arcface = ArcFaceLayer(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )

    def forward(self, x, labels):
        features = self.backbone(x)
        embeddings = self.embedding_net(features)
        logits = self.arcface(embeddings, labels)
        return logits, embeddings

    def get_embeddings(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_net(features)
        return F.normalize(embeddings, p=2, dim=1)


class ArcFaceHeadModel(nn.Module):
    """Projection + ArcFace head for cached backbone embeddings."""

    def __init__(
        self,
        input_dim,
        num_classes,
        embedding_dim=256,
        hidden_dim=512,
        margin=0.5,
        scale=64.0,
        dropout=0.3,
    ):
        super().__init__()
        self.embedding_net = EmbeddingProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        self.arcface = ArcFaceLayer(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )

    def forward(self, embeddings, labels):
        projected = self.embedding_net(embeddings)
        logits = self.arcface(projected, labels)
        return logits, projected

    def get_embeddings(self, embeddings):
        projected = self.embedding_net(embeddings)
        return F.normalize(projected, p=2, dim=1)
