import numpy as np
import torch
import torch.nn.functional as F


# ---------- cosine similarity ----------
def cosine_similarity(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    return F.cosine_similarity(a, b, dim=0).item()


# ---------- retrieve top-k ----------
def retrieve_top_k(query_emb, db_embs, k=5):

    similarities = []

    for i in range(len(db_embs)):
        sim = cosine_similarity(query_emb, db_embs[i])
        similarities.append((i, float(sim)))  # store index + score

    # sort descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # separate index & score
    topk_idx = [x[0] for x in similarities[:k]]
    topk_scores = [x[1] for x in similarities[:k]]

    return topk_idx, topk_scores
