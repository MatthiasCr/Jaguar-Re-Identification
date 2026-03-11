import numpy as np


def cosine_similarity_matrix(query_embeddings, gallery_embeddings=None):
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    if gallery_embeddings is None:
        gallery_embeddings = query_embeddings
    else:
        gallery_embeddings = np.asarray(gallery_embeddings, dtype=np.float32)

    sim_matrix = query_embeddings @ gallery_embeddings.T
    return np.clip(sim_matrix, -1.0, 1.0)


def squared_euclidean_distance_from_cosine(query_embeddings, gallery_embeddings=None):
    sim_matrix = cosine_similarity_matrix(query_embeddings, gallery_embeddings)
    return np.clip(2.0 - 2.0 * sim_matrix, 0.0, None).astype(np.float32)


def k_reciprocal_rerank(q_g_dist, q_q_dist=None, g_g_dist=None, k1=20, k2=6, lambda_value=0.3):
    q_g_dist = np.asarray(q_g_dist, dtype=np.float32)
    query_num, gallery_num = q_g_dist.shape

    if q_q_dist is None:
        q_q_dist = np.zeros((query_num, query_num), dtype=np.float32)
    else:
        q_q_dist = np.asarray(q_q_dist, dtype=np.float32)

    if g_g_dist is None:
        g_g_dist = np.zeros((gallery_num, gallery_num), dtype=np.float32)
    else:
        g_g_dist = np.asarray(g_g_dist, dtype=np.float32)

    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1),
        ],
        axis=0,
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)

    col_max = np.max(original_dist, axis=0)
    col_max[col_max == 0] = 1.0
    original_dist = np.transpose(original_dist / col_max)

    all_num = query_num + gallery_num
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    for i in range(all_num):
        forward = initial_rank[i, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        fi = np.where(backward == i)[0]
        reciprocal = forward[fi]

        reciprocal_expansion = reciprocal.copy()
        half_k = int(np.around(k1 / 2)) + 1
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, :half_k]
            candidate_backward = initial_rank[candidate_forward, :half_k]
            fi_candidate = np.where(candidate_backward == candidate)[0]
            candidate_reciprocal = candidate_forward[fi_candidate]

            if len(candidate_reciprocal) == 0:
                continue

            overlap = np.intersect1d(candidate_reciprocal, reciprocal)
            if len(overlap) > (2.0 / 3.0) * len(candidate_reciprocal):
                reciprocal_expansion = np.append(reciprocal_expansion, candidate_reciprocal)

        reciprocal_expansion = np.unique(reciprocal_expansion)
        weights = np.exp(-original_dist[i, reciprocal_expansion])
        V[i, reciprocal_expansion] = weights / np.sum(weights)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    original_dist = original_dist[:query_num, :]
    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        non_zero = np.where(V[i, :] != 0)[0]
        non_zero_images = [inv_index[index] for index in non_zero]
        for j, index in enumerate(non_zero):
            temp_min[0, non_zero_images[j]] += np.minimum(V[i, index], V[non_zero_images[j], index])
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min)

    final_dist = (1.0 - lambda_value) * jaccard_dist + lambda_value * original_dist
    return final_dist[:, query_num:]


def rerank_embeddings(query_embeddings, gallery_embeddings=None, k1=20, k2=6, lambda_value=0.3):
    if gallery_embeddings is None:
        gallery_embeddings = query_embeddings

    q_g_dist = squared_euclidean_distance_from_cosine(query_embeddings, gallery_embeddings)
    q_q_dist = squared_euclidean_distance_from_cosine(query_embeddings, query_embeddings)
    g_g_dist = squared_euclidean_distance_from_cosine(gallery_embeddings, gallery_embeddings)

    return k_reciprocal_rerank(
        q_g_dist,
        q_q_dist=q_q_dist,
        g_g_dist=g_g_dist,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )


def reranked_similarity_from_embeddings(query_embeddings, gallery_embeddings=None, k1=20, k2=6, lambda_value=0.3):
    final_dist = rerank_embeddings(
        query_embeddings,
        gallery_embeddings=gallery_embeddings,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    return np.clip(1.0 - final_dist, 0.0, 1.0)
