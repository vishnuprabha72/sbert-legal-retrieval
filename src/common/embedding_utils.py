"""Thin wrapper around SentenceTransformer so every stage embeds consistently."""
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer, util


@lru_cache(maxsize=4)
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str) -> np.ndarray:
    model = get_model(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, d), b: (m, d), both pre-normalized -> (n, m) similarity matrix."""
    return a @ b.T


def best_match(query_emb: np.ndarray, candidate_embs: np.ndarray) -> tuple[int, float]:
    sims = candidate_embs @ query_emb
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])
