import numpy as np
import torch
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
