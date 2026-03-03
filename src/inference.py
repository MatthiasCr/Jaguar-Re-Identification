import numpy as np
import torch
import pandas as pd
from tqdm.notebook import tqdm


def extract_embeddings_with_names(model, loader, device):
    model.eval()
    embeddings = []
    names = []

    with torch.no_grad():
        for images, batch_names in tqdm(loader, desc="Embedding", leave=False):
            images = images.to(device)
            batch_emb = model.get_embeddings(images).cpu().numpy()
            embeddings.append(batch_emb)
            names.extend(batch_names)

    embeddings = np.vstack(embeddings)
    return names, embeddings


def build_embedding_lookup(names, embeddings):
    return {name: emb for name, emb in zip(names, embeddings)}


def compute_similarity_for_pairs(pairs_df, embedding_lookup):
    similarities = []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Computing similarities"):
        query_emb = embedding_lookup[row["query_image"]]
        gallery_emb = embedding_lookup[row["gallery_image"]]
        sim = float(np.dot(query_emb, gallery_emb))
        similarities.append(sim)

    similarities = np.array(similarities)
    similarities = np.clip(similarities, 0.0, 1.0)
    return similarities


def extract_embeddings_with_names_backbone(backbone, head_model, loader, device):
    backbone.eval()
    head_model.eval()
    embeddings = []
    names = []

    with torch.no_grad():
        for images, batch_names in tqdm(loader, desc="Embedding", leave=False):
            images = images.to(device)
            features = backbone(images)
            batch_emb = head_model.get_embeddings(features).cpu().numpy()
            embeddings.append(batch_emb)
            names.extend(batch_names)

    embeddings = np.vstack(embeddings)
    return names, embeddings



def create_submission_backbone(backbone, head_model, device, pairs_df, test_loader, output_path=None):
    names, embeddings = extract_embeddings_with_names_backbone(backbone, head_model, test_loader, device)
    embedding_lookup = build_embedding_lookup(names, embeddings)
    similarities = compute_similarity_for_pairs(pairs_df, embedding_lookup)
    submission_df = pd.DataFrame(
        {
            "row_id": pairs_df["row_id"],
            "similarity": similarities,
        }
    )
    if output_path is not None:
        submission_df.to_csv(output_path, index=False)
    return submission_df