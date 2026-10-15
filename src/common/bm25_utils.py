"""
BM25 lexical scoring + Reciprocal Rank Fusion (RRF) with cosine similarity.

Why both signals:
  - Cosine (dense embeddings) is good at paraphrase / semantic equivalence
    (e.g. "the tenant kept possession" ~ "possession was not parted with").
  - BM25 (lexical, term-overlap) is good at exact legal tokens embeddings
    tend to underweight: section numbers ("Section 2(6)"), case citations
    ("AIR 1968 SC 933"), party/statute names.

RRF combines two independently-computed RANKINGS (not raw scores) so we
never have to normalize BM25's unbounded scale against cosine's [0,1]
scale, and it avoids the fragility of hand-picked score weights.

    RRF(d) = sum_over_rankers( 1 / (k + rank_of_d_in_that_ranker) )

k=60 is the standard constant from the original RRF paper (Cormack et al.)
and is not sensitive to tuning across a wide range of corpus sizes.
"""
import re
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def build_bm25(candidate_texts: Sequence[str]) -> BM25Okapi:
    corpus_tokens = [tokenize(t) for t in candidate_texts]
    return BM25Okapi(corpus_tokens)


def bm25_scores_for_query(bm25: BM25Okapi, query_text: str) -> np.ndarray:
    return np.asarray(bm25.get_scores(tokenize(query_text)), dtype=float)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Returns 1-based rank of each element (rank 1 = highest score)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf_fuse(cosine_scores: np.ndarray, bm25_scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Fuse two score arrays (same candidate order) into one RRF score per
    candidate, purely from their rank positions within each ranker."""
    cos_ranks = ranks_from_scores(cosine_scores)
    bm25_ranks = ranks_from_scores(bm25_scores)
    return 1.0 / (k + cos_ranks) + 1.0 / (k + bm25_ranks)
