import numpy as np
import torch
import pandas as pd
from tqdm.notebook import tqdm
from src.reranking import cosine_similarity_matrix, reranked_similarity_from_embeddings


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


def build_similarity_matrix(image_names, embedding_lookup):
    embeddings = np.vstack([embedding_lookup[name] for name in image_names]).astype(np.float32)
    return cosine_similarity_matrix(embeddings), embeddings


def rerank_similarity_matrix(embeddings, enabled=False, k1=20, k2=6, lambda_value=0.3):
    if not enabled:
        return np.clip(cosine_similarity_matrix(embeddings), 0.0, 1.0)

    return reranked_similarity_from_embeddings(
        embeddings,
        gallery_embeddings=embeddings,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )


def compute_similarity_for_pairs(pairs_df, embedding_lookup, use_rerank=False, k1=20, k2=6, lambda_value=0.3):
    image_names = sorted(set(pairs_df["query_image"]) | set(pairs_df["gallery_image"]))
    sim_matrix, embeddings = build_similarity_matrix(image_names, embedding_lookup)
    pd.DataFrame(sim_matrix, index=image_names, columns=image_names).to_csv("similarity_matrix_before_rerank.csv")

    sim_matrix = rerank_similarity_matrix(
        embeddings,
        enabled=use_rerank,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    pd.DataFrame(sim_matrix, index=image_names, columns=image_names).to_csv("similarity_matrix_after_rerank.csv")

    image_to_index = {name: idx for idx, name in enumerate(image_names)}
    similarities = np.empty(len(pairs_df), dtype=np.float32)

    for row_idx, row in enumerate(tqdm(pairs_df.itertuples(index=False), total=len(pairs_df), desc="Building submission scores")):
        query_idx = image_to_index[row.query_image]
        gallery_idx = image_to_index[row.gallery_image]
        similarities[row_idx] = sim_matrix[query_idx, gallery_idx]

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


def create_submission_model(
    model,
    device,
    pairs_df,
    test_loader,
    output_path=None,
    use_rerank=False,
    k1=20,
    k2=6,
    lambda_value=0.3,
):
    names, embeddings = extract_embeddings_with_names(model, test_loader, device)
    embedding_lookup = build_embedding_lookup(names, embeddings)
    similarities = compute_similarity_for_pairs(
        pairs_df,
        embedding_lookup,
        use_rerank=use_rerank,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    submission_df = pd.DataFrame(
        {
            "row_id": pairs_df["row_id"],
            "similarity": similarities,
        }
    )
    if output_path is not None:
        submission_df.to_csv(output_path, index=False)
    return submission_df


def create_submission_backbone(
    backbone,
    head_model,
    device,
    pairs_df,
    test_loader,
    output_path=None,
    use_rerank=False,
    k1=20,
    k2=6,
    lambda_value=0.3,
):
    names, embeddings = extract_embeddings_with_names_backbone(backbone, head_model, test_loader, device)
    embedding_lookup = build_embedding_lookup(names, embeddings)
    similarities = compute_similarity_for_pairs(
        pairs_df,
        embedding_lookup,
        use_rerank=use_rerank,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    submission_df = pd.DataFrame(
        {
            "row_id": pairs_df["row_id"],
            "similarity": similarities,
        }
    )
    if output_path is not None:
        submission_df.to_csv(output_path, index=False)
    return submission_df
