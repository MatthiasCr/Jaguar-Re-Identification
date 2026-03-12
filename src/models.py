import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def build_backbone(backbone_name: str, pretrained: bool = True):
    return timm.create_model(backbone_name, pretrained=pretrained
    # , features_only=True
    , num_classes=0
    )


class GeM(nn.Module):
    """Generalized mean pooling over the spatial dimensions."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)


class GeMBackbone(nn.Module):
    """Wrap a timm backbone and replace its default pooling with GeM."""

    def __init__(self, backbone_name: str, pretrained: bool = True, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.gem = GeM(p=p, eps=eps)
        self.num_features = getattr(self.backbone, "num_features", None)
        if self.num_features is None:
            raise ValueError("Backbone output dimension not found.")

    def _tokens_to_grid(self, features):
        if features.dim() != 3:
            return features

        num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 0)
        tokens = features[:, num_prefix_tokens:, :]
        batch_size, num_tokens, channels = tokens.shape

        grid_size = getattr(getattr(self.backbone, "patch_embed", None), "grid_size", None)
        if grid_size is not None:
            height, width = grid_size
        else:
            side = int(math.sqrt(num_tokens))
            if side * side != num_tokens:
                raise ValueError(f"Cannot infer token grid from {num_tokens} tokens.")
            height = width = side

        if height * width != num_tokens:
            raise ValueError(f"Token grid mismatch: grid {height}x{width} != {num_tokens} tokens.")

        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if features.dim() == 2:
            return features

        features = self._tokens_to_grid(features)
        return self.gem(features).flatten(1)


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


class CosFaceLayer(nn.Module):
    """CosFace (additive cosine margin) layer."""

    def __init__(self, embedding_dim, num_classes, margin=0.35, scale=64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight_norm).clamp(-1.0, 1.0)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = cosine - one_hot * self.margin
        output = output * self.scale
        return output


class SphereFaceLayer(nn.Module):
    """SphereFace (multiplicative angular margin) layer."""

    def __init__(self, embedding_dim, num_classes, margin=1.35, scale=64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        theta = torch.acos(cosine)
        phi = torch.cos(self.margin * theta)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.scale
        return output


class LinearClassifier(nn.Module):
    """Plain linear classifier logits for cross-entropy."""

    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings, labels):
        del labels
        return self.classifier(embeddings)


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
        freeze_backbone=False,
        train_last_n_layers=0,
        freeze_last_n_layers=None,
        use_gem=False,
        gem_p=3.0,
        gem_eps=1e-6,
    ):
        super().__init__()
        if use_gem:
            self.backbone = GeMBackbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                p=gem_p,
                eps=gem_eps,
            )
        else:
            self.backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.set_backbone_trainable(True)
        if freeze_backbone:
            self.set_backbone_trainable(False)
        elif train_last_n_layers > 0:
            self.freeze_backbone_all_but_last_n_layers(train_last_n_layers)
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

    def set_backbone_trainable(self, trainable: bool = True):
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def freeze_backbone_all_but_last_n_layers(self, train_last_n_layers: int):
        if train_last_n_layers < 0:
            raise ValueError("train_last_n_layers must be >= 0")

        self.set_backbone_trainable(False)
        if train_last_n_layers == 0:
            return

        if hasattr(self.backbone, "blocks"):
            blocks = list(self.backbone.blocks)
            num_blocks_to_train = min(train_last_n_layers, len(blocks))
            for block in blocks[-num_blocks_to_train:]:
                for param in block.parameters():
                    param.requires_grad = True

            if num_blocks_to_train > 0:
                for attr_name in ("norm", "fc_norm"):
                    if hasattr(self.backbone, attr_name):
                        for param in getattr(self.backbone, attr_name).parameters():
                            param.requires_grad = True
            return

        param_groups = []
        for _, module in self.backbone.named_children():
            params = list(module.parameters())
            if params:
                param_groups.append(params)

        if not param_groups:
            return

        num_groups_to_train = min(train_last_n_layers, len(param_groups))
        for params in param_groups[-num_groups_to_train:]:
            for param in params:
                param.requires_grad = True

    def freeze_backbone_last_n_layers(self, freeze_last_n_layers: int):
        # Backward-compatible alias; the intended semantics are "train last n layers".
        self.freeze_backbone_all_but_last_n_layers(freeze_last_n_layers)

    def forward(self, x, labels):
        features = self.backbone(x)
        embeddings = self.embedding_net(features)
        logits = self.arcface(embeddings, labels)
        return logits, embeddings

    def get_embeddings(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_net(features)
        return F.normalize(embeddings, p=2, dim=1)


def load_arcface_model_from_checkpoint(checkpoint_path, device, strict: bool = True):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint["config"]

    num_classes = checkpoint.get("num_classes")
    if num_classes is None:
        num_classes = len(checkpoint["label_encoder_classes"])

    model = ArcFaceModel(
        backbone_name=checkpoint_config["backbone_name"],
        num_classes=num_classes,
        embedding_dim=int(checkpoint_config["embedding_dim"]),
        hidden_dim=int(checkpoint_config["hidden_dim"]),
        margin=float(checkpoint_config.get("arcface_margin", 0.5)),
        scale=float(checkpoint_config.get("arcface_scale", 64.0)),
        dropout=float(checkpoint_config.get("dropout", 0.3)),
        pretrained=False,
        freeze_backbone=False,
        train_last_n_layers=0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.eval()
    return model, checkpoint, checkpoint_config


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


class MetricLearningModel(nn.Module):
    """Projection + arcface/cosface/sphereface head for cached backbone embeddings."""

    def __init__(
        self,
        input_dim,
        num_classes,
        loss_name="arcface",
        embedding_dim=256,
        hidden_dim=512,
        arcface_margin=0.5,
        cosface_margin=0.35,
        sphereface_margin=1.35,
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
        if loss_name == "arcface":
            self.head = ArcFaceLayer(embedding_dim, num_classes, margin=arcface_margin,scale=scale)
        elif loss_name == "cosface":
            self.head = CosFaceLayer(embedding_dim, num_classes, margin=cosface_margin, scale=scale)
        elif loss_name == "sphereface":
            self.head = SphereFaceLayer(embedding_dim, num_classes, margin=sphereface_margin, scale=scale)
        elif loss_name == "ce":
            self.head = LinearClassifier(embedding_dim, num_classes)

    def forward(self, embeddings, labels):
        projected = self.embedding_net(embeddings)
        logits = self.head(projected, labels)
        return logits, projected

    def get_embeddings(self, embeddings):
        projected = self.embedding_net(embeddings)
        return F.normalize(projected, p=2, dim=1)


class FocalLoss(nn.Module):
    """Multi-class focal loss on logits."""

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([float(alpha)])
        else:
            self.alpha = torch.as_tensor(alpha, dtype=torch.float)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            if alpha.numel() == 1:
                focal_weight = focal_weight * alpha
            else:
                focal_weight = focal_weight * alpha[targets]

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
