from pathlib import Path

import numpy as np
import torch
from tqdm.notebook import tqdm


def extract_backbone_embeddings(backbone, loader, device):
    backbone.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Backbone", leave=False):
            images = images.to(device)
            emb = backbone(images).cpu().numpy()
            embeddings_list.append(emb)
            labels_list.append(labels.numpy())

    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    return embeddings, labels


def save_embeddings(path, embeddings, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, labels=labels)


def load_embeddings(path):
    z = np.load(path, allow_pickle=True)
    return z["embeddings"], z["labels"]


def get_or_create_embeddings(path, backbone, loader, device):
    path = Path(path)
    if path.exists():
        return load_embeddings(path)

    embeddings, labels = extract_backbone_embeddings(backbone, loader, device)
    save_embeddings(path, embeddings, labels)
    return embeddings, labels
