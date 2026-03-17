from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


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


def load_image_tensor(filename, image_dir, transform):
    image = Image.open(Path(image_dir) / filename).convert("RGB")
    tensor = transform(image)
    return image, tensor


def compute_attribution_result(
    filename,
    classifier,
    integrated_gradients_cls,
    label_encoder,
    device,
    image_dir,
    transform,
    target_idx=None,
):
    image, tensor = load_image_tensor(filename, image_dir=image_dir, transform=transform)
    input_batch = tensor.unsqueeze(0).to(device)
    input_batch.requires_grad_(True)

    logits = classifier(input_batch)
    if target_idx is None:
        target_idx = int(logits.argmax(dim=1).item())

    ig = integrated_gradients_cls(classifier)
    attribution = ig.attribute(
        input_batch,
        target=target_idx,
        baselines=torch.zeros_like(input_batch),
    )
    heatmap = attribution_to_heatmap(attribution)[0].detach().cpu().numpy()
    return {
        "filename": filename,
        "image": image,
        "heatmap": heatmap,
        "pred_idx": target_idx,
        "pred_label": label_encoder.inverse_transform([target_idx])[0],
        "pred_score": float(logits[0, target_idx].item()),
    }


def prepare_display_heatmap(result, upper_percentile=99.9):
    heatmap = result["heatmap"]
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)).resize(
        result["image"].size,
        resample=Image.BILINEAR,
    )
    heatmap_resized = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    upper = np.percentile(heatmap_resized, upper_percentile)
    if upper > 0:
        heatmap_resized = np.clip(heatmap_resized / upper, 0.0, 1.0)

    return heatmap_resized


def build_top_mask(heatmap, top_fraction):
    flat = heatmap.reshape(-1)
    k = max(1, int(len(flat) * top_fraction))
    threshold = np.partition(flat, -k)[-k]
    return heatmap >= threshold


def apply_mask_to_tensor(tensor, mask, fill_value=0.0):
    masked = tensor.clone()
    mask_tensor = torch.from_numpy(mask).to(masked.device)
    masked[:, mask_tensor] = fill_value
    return masked


def apply_random_mask_to_tensor(tensor, num_pixels, rng, fill_value=0.0):
    masked = tensor.clone()
    flat_mask = np.zeros(masked.shape[-2] * masked.shape[-1], dtype=bool)
    random_indices = rng.choice(len(flat_mask), size=num_pixels, replace=False)
    flat_mask[random_indices] = True
    random_mask = flat_mask.reshape(masked.shape[-2], masked.shape[-1])
    masked[:, torch.from_numpy(random_mask).to(masked.device)] = fill_value
    return masked, random_mask


def build_similarity_pairs(val_examples, num_true_pairs=5, num_impostor_pairs=5):
    correct_examples = val_examples[val_examples["is_correct"]].copy()

    true_pairs = []
    for _, group in correct_examples.groupby("ground_truth"):
        if len(group) < 2:
            continue
        group = group.sort_values(["pred_score", "filename"], ascending=[False, True]).head(2)
        query_row, gallery_row = group.iloc[0], group.iloc[1]
        true_pairs.append(
            {
                "pair_type": "true_match",
                "query_filename": query_row["filename"],
                "gallery_filename": gallery_row["filename"],
                "query_label": query_row["ground_truth"],
                "gallery_label": gallery_row["ground_truth"],
                "target_idx": int(query_row["pred_idx"]),
            }
        )
        if len(true_pairs) >= num_true_pairs:
            break

    impostor_pairs = []
    sorted_examples = correct_examples.sort_values(
        ["pred_score", "filename"], ascending=[False, True]
    ).reset_index(drop=True)
    for idx, query_row in sorted_examples.iterrows():
        for _, gallery_row in sorted_examples.iloc[idx + 1 :].iterrows():
            if query_row["ground_truth"] == gallery_row["ground_truth"]:
                continue
            impostor_pairs.append(
                {
                    "pair_type": "impostor",
                    "query_filename": query_row["filename"],
                    "gallery_filename": gallery_row["filename"],
                    "query_label": query_row["ground_truth"],
                    "gallery_label": gallery_row["ground_truth"],
                    "target_idx": int(query_row["pred_idx"]),
                }
            )
            break
        if len(impostor_pairs) >= num_impostor_pairs:
            break

    return pd.DataFrame(true_pairs + impostor_pairs)
