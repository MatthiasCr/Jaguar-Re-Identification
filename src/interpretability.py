from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceClassifierForLRP(nn.Module):
    def __init__(self, arcface_model):
        super().__init__()
        self.arcface_model = arcface_model

    def forward(self, images):
        features = self.arcface_model.backbone(images)
        embeddings = self.arcface_model.embedding_net(features)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_weight = F.normalize(self.arcface_model.arcface.weight, p=2, dim=1)
        return F.linear(normalized_embeddings, normalized_weight)


def cosine_logits_from_arcface(head_model, features):
    projected = head_model.embedding_net(features)
    normalized_projected = F.normalize(projected, p=2, dim=1)
    normalized_weight = F.normalize(head_model.arcface.weight, p=2, dim=1)
    logits = F.linear(normalized_projected, normalized_weight)
    return logits, projected


def epsilon_lrp_linear(x, layer, relevance_out, eps=1e-6):
    weight = layer.weight
    bias = layer.bias if layer.bias is not None else 0.0

    z = F.linear(x, weight, bias)
    stabilizer = eps * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    s = relevance_out / (z + stabilizer)
    c = torch.matmul(s, weight)
    return x * c


def epsilon_lrp_batchnorm1d(x, layer, relevance_out, eps=1e-6):
    if layer.training:
        raise ValueError("BatchNorm LRP expects eval mode.")

    scale = layer.weight / torch.sqrt(layer.running_var + layer.eps)
    bias = layer.bias - layer.running_mean * scale
    z = x * scale + bias
    stabilizer = eps * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    s = relevance_out / (z + stabilizer)
    c = s * scale
    return x * c


def epsilon_lrp_embedding_net(embedding_net, inputs, relevance_out, eps=1e-6):
    modules = list(embedding_net.network.children())
    activations = [inputs]
    x = inputs

    for module in modules:
        x = module(x)
        activations.append(x)

    relevance = relevance_out
    for idx in reversed(range(len(modules))):
        module = modules[idx]
        x_in = activations[idx]

        if isinstance(module, nn.Linear):
            relevance = epsilon_lrp_linear(x_in, module, relevance, eps=eps)
        elif isinstance(module, nn.BatchNorm1d):
            relevance = epsilon_lrp_batchnorm1d(x_in, module, relevance, eps=eps)
        elif isinstance(module, (nn.ReLU, nn.Dropout)):
            continue
        else:
            raise TypeError(f"Unsupported module for epsilon-LRP: {type(module).__name__}")

    return relevance


def compute_head_feature_relevance(head_model, pooled_features, class_idx=None, eps=1e-6):
    pooled_features = pooled_features.clone().detach()
    pooled_features.requires_grad_(True)

    logits, projected = cosine_logits_from_arcface(head_model, pooled_features)
    if class_idx is None:
        class_idx = logits.argmax(dim=1)
    elif not torch.is_tensor(class_idx):
        class_idx = torch.tensor([class_idx], device=logits.device)

    target_logits = logits.gather(1, class_idx.view(-1, 1)).sum()
    projected_grad = torch.autograd.grad(target_logits, projected, retain_graph=False)[0]
    projected_relevance = projected * projected_grad

    feature_relevance = epsilon_lrp_embedding_net(
        head_model.embedding_net,
        pooled_features,
        projected_relevance.detach(),
        eps=eps,
    )
    return {
        "feature_relevance": feature_relevance.detach(),
        "logits": logits.detach(),
        "class_idx": class_idx.detach(),
    }


def project_feature_relevance_to_map(feature_map, feature_relevance):
    if feature_map.ndim != 4:
        raise ValueError("feature_map must be a 4D tensor [B, C, H, W].")
    if feature_relevance.ndim != 2:
        raise ValueError("feature_relevance must be a 2D tensor [B, C].")

    positive_maps = feature_map.clamp(min=0)
    channel_mass = positive_maps.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    channel_weights = feature_relevance[:, :, None, None]
    projected = positive_maps * channel_weights / channel_mass
    heatmap = projected.sum(dim=1)
    heatmap = heatmap.clamp(min=0)

    flat = heatmap.flatten(start_dim=1)
    mins = flat.min(dim=1)[0][:, None, None]
    maxs = flat.max(dim=1)[0][:, None, None]
    heatmap = (heatmap - mins) / (maxs - mins).clamp_min(1e-8)
    return heatmap


def upsample_heatmap(heatmap, size):
    if heatmap.ndim == 2:
        heatmap = heatmap[None, None, :, :]
    elif heatmap.ndim == 3:
        heatmap = heatmap[:, None, :, :]

    return F.interpolate(
        heatmap,
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def normalize_heatmap(heatmap):
    if heatmap.ndim == 3:
        flat = heatmap.flatten(start_dim=1)
        mins = flat.min(dim=1)[0][:, None, None]
        maxs = flat.max(dim=1)[0][:, None, None]
        return (heatmap - mins) / (maxs - mins).clamp_min(1e-8)

    if heatmap.ndim == 2:
        min_value = heatmap.min()
        max_value = heatmap.max()
        return (heatmap - min_value) / (max_value - min_value).clamp_min(1e-8)

    raise ValueError("Expected a 2D or 3D heatmap tensor.")


def lrp_attribution_to_heatmap(attribution):
    if attribution.ndim != 4:
        raise ValueError("Expected attribution tensor of shape [B, C, H, W].")

    heatmap = attribution.abs().sum(dim=1)
    return normalize_heatmap(heatmap)
