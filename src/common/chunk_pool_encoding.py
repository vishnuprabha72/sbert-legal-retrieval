"""
Training-time (differentiable) chunk-and-pool encoding, and a matching
no-grad evaluator for validation/model-selection during training.

Why: sentence-transformers' legacy `.fit()` wrapper tokenizes each
training text with truncation=True, max_length=model.max_seq_length (256
tokens for all-MiniLM-L6-v2). We checked this by appending a nonsense
suffix to a long judgment: the embedding did not change. Anything past
token 256 never reaches the transformer during training. For legal text (denser tokenization than plain English; median
segment ~274 tokens even under the 300-word merge cap), that
throws away a meaningful chunk of many training paragraphs' content,
in every epoch.

Approach: for any paragraph over the token limit, split it into
sentence-aware, token-bounded chunks (chunk_text_by_tokens), run EACH
chunk through the model with gradients attached, and mean-pool the
resulting chunk embeddings into one training-time embedding for that
paragraph. This is done for BOTH sides of every training pair (query and
positive) for uniformity, though in practice queries are short (5-40
words per config.yaml's word_limits) and almost always resolve to a
single chunk -- the chunking only changes behavior for text that actually
exceeds the limit.

WHY model.fit() COULDN'T BE REUSED FOR THIS: model.fit()'s tokenization
and truncation happen inside smart_batching_collate, before any loss
function ever sees the text -- there's no hook to intercept a single
oversized text, split it, encode the pieces, and pool them back into one
embedding before the loss is computed. That's why stage5_train_sbert.py
now runs a plain manual training loop (optimizer/scheduler/backward
managed directly) instead of model.fit() when chunk_pool.enabled=true in
config.yaml, using ChunkPoolMultipleNegativesRankingLoss below in place
of sentence-transformers' built-in MultipleNegativesRankingLoss.

WHY VALIDATION NEEDS THE SAME TREATMENT: sentence-transformers'
InformationRetrievalEvaluator calls model.encode() directly on the val
corpus, which would silently re-truncate any long val-set positive again
-- undoing the point of chunk-pool training for exactly the metric used
to pick the "best" epoch. evaluate_retrieval_chunk_pool below embeds the
val corpus/queries with the identical chunk-and-pool method (no_grad),
so "best epoch by NDCG@10" is judged consistently with how the model was
trained.

COST TRADEOFF (documented, not hidden): a paragraph needing 3 chunks
costs ~3x the forward passes of a single truncated pass. On this
project's CPU-only hardware that measurably slows both training and
validation. Used anyway because plain truncation was PROVEN (not
assumed) to lose the great majority of a long judgment's content.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.text_utils import chunk_text_by_tokens

# Small safety margin: chunk_text_by_tokens counts tokens with
# add_special_tokens=False, but model.tokenize() adds [CLS]/[SEP] (or
# equivalent) back on top per chunk. Without this margin a chunk sized
# to exactly max_seq_length could still get silently re-truncated by a
# couple of tokens once special tokens are added back -- small, but
# defeats the purpose if left in.
SPECIAL_TOKEN_MARGIN = 2

# HARD CAP on chunks per individual text (found necessary in practice, not
# theoretical): encode_with_chunk_pool flattens EVERY row's chunks in a
# batch into ONE forward pass -- so a single outlier-length row (e.g. an
# extreme judgment or summary sentence that slipped past Stage 0's 300-word
# cap, which only ever applied to judgment_original segments, never to
# llm_generated summary sentences) can make that one training step's total
# chunk count -- and therefore its CPU cost -- balloon far beyond what
# batch_size alone would suggest, with no natural upper bound. At
# MAX_CHUNKS_PER_TEXT=20 (~20*254 ≈ 5000 tokens ≈ ~3500-4000 words per
# text), a text beyond this is truncated at the chunk level -- still
# capturing roughly the first 3500-4000 words instead of ~150-200 (the
# original 256-token truncation), i.e. still a large improvement over the
# flaw this whole mechanism exists to fix, just with a bounded worst case
# instead of an unbounded one. A text hitting this cap is logged once via
# _log_chunk_cap_hit so outliers are visible rather than silently capped.
MAX_CHUNKS_PER_TEXT = 20

_chunk_cap_warned: set[str] = set()


def _log_chunk_cap_hit(text: str, n_chunks_before_cap: int) -> None:
    key = text[:80]
    if key in _chunk_cap_warned:
        return
    _chunk_cap_warned.add(key)
    import sys
    print(f"  [chunk_pool] NOTE: a text needed {n_chunks_before_cap} chunks (> cap of "
          f"{MAX_CHUNKS_PER_TEXT}) -- capped to the first {MAX_CHUNKS_PER_TEXT} chunks "
          f"(~{MAX_CHUNKS_PER_TEXT * 250} words). Text starts: {text[:80]!r}...", file=sys.stderr)


def _effective_chunk_budget(model, max_tokens: int = None) -> int:
    max_tokens = max_tokens or model.max_seq_length
    return max(max_tokens - SPECIAL_TOKEN_MARGIN, 1)


def _pool_chunks(chunk_embs: torch.Tensor, row_of_chunk: list[int], n_rows: int, pooling: str) -> torch.Tensor:
    """
    Pools variable-count chunk embeddings back down to one vector per
    original text, gradients intact. Implemented as a per-row loop
    (not a vectorized scatter) deliberately -- chunk counts per
    paragraph are small in practice (usually 1-2 at this project's
    token limit), so the loop costs nothing meaningful, and it keeps
    both "mean" and "max" pooling equally simple and correct without
    depending on torch-version-specific scatter_reduce behavior.
    """
    pooled = []
    for row in range(n_rows):
        idx = [i for i, r in enumerate(row_of_chunk) if r == row]
        row_embs = chunk_embs[idx]
        if pooling == "mean":
            pooled.append(row_embs.mean(dim=0))
        elif pooling == "max":
            pooled.append(row_embs.max(dim=0).values)
        else:
            raise ValueError(f"Unknown pooling method: {pooling!r} (expected 'mean' or 'max')")
    pooled = torch.stack(pooled, dim=0)
    # re-normalize -- mean/max of unit vectors isn't itself unit-length,
    # and MultipleNegativesRankingLoss below assumes cosine-normalized inputs
    return F.normalize(pooled, p=2, dim=1)


def encode_with_chunk_pool(model, texts: list[str], max_tokens: int = None, pooling: str = "mean") -> torch.Tensor:
    """
    Differentiable chunk-and-pool encoding for a BATCH of texts.
    Gradients flow through this end-to-end: chunk -> tokenize -> model
    forward (no no_grad) -> pool. For any text under the token budget
    this reduces to exactly one "chunk" == the whole text, i.e. behaves
    identically to a normal single forward pass for short text.
    """
    budget = _effective_chunk_budget(model, max_tokens)
    tokenizer = model.tokenizer

    all_chunks: list[str] = []
    row_of_chunk: list[int] = []
    for row_idx, text in enumerate(texts):
        chunks = chunk_text_by_tokens(text, tokenizer, budget)
        if not chunks:
            chunks = [text or ""]
        if len(chunks) > MAX_CHUNKS_PER_TEXT:
            _log_chunk_cap_hit(text or "", len(chunks))
            chunks = chunks[:MAX_CHUNKS_PER_TEXT]
        all_chunks.extend(chunks)
        row_of_chunk.extend([row_idx] * len(chunks))

    features = model.tokenize(all_chunks)
    # NOTE: newer sentence-transformers versions (confirmed on 5.6.0) include
    # non-tensor metadata in this dict alongside the actual tensors -- e.g. a
    # "modality" key whose value is a plain string like "text", and
    # optionally "prompt_length". Blindly calling .to(device) on every value
    # crashes on those with `AttributeError: 'str' object has no attribute
    # 'to'`. Only move actual tensors; pass everything else through as-is so
    # the model's forward pass still receives it (it expects "modality" back).
    features = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in features.items()}
    out = model(features)                    # forward WITH gradients -- no torch.no_grad() here
    chunk_embs = out["sentence_embedding"]    # [total_chunks, dim]

    return _pool_chunks(chunk_embs, row_of_chunk, len(texts), pooling)


class ChunkPoolMultipleNegativesRankingLoss(nn.Module):
    """
    Same objective as sentence-transformers' MultipleNegativesRankingLoss
    (in-batch negatives: cosine similarity matrix x scale, cross-entropy
    against the diagonal) but encodes BOTH the query and the positive via
    encode_with_chunk_pool instead of a single truncated forward pass --
    so a long positive paragraph's full content contributes gradient
    signal, not just whatever fit in the first ~256 tokens.

    OPTIONAL explicit hard negatives (forward(..., negatives=[...])):
    when provided, each row's mined hard negative is appended as an
    EXTRA candidate column alongside the in-batch ones -- candidate pool
    becomes [all positives in batch; all hard negatives in batch], shape
    [2*B, dim], scored against the B queries, cross-entropy still against
    the diagonal (each query's true positive stays at column i). This
    matches how sentence-transformers' own MultipleNegativesRankingLoss
    treats a 3-text (anchor, positive, negative) InputExample -- the
    explicit negative augments in-batch negatives, it doesn't replace
    them. Passing negatives=None reproduces the original in-batch-only
    behavior exactly (used for the ablation comparison in
    ablation_negatives.py).
    """
    def __init__(self, model, max_tokens: int = None, pooling: str = "mean", scale: float = 20.0):
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.pooling = pooling
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, queries: list[str], positives: list[str], negatives: list[str] = None) -> torch.Tensor:
        q_emb = encode_with_chunk_pool(self.model, queries, self.max_tokens, self.pooling)
        p_emb = encode_with_chunk_pool(self.model, positives, self.max_tokens, self.pooling)

        if negatives is not None:
            n_emb = encode_with_chunk_pool(self.model, negatives, self.max_tokens, self.pooling)
            candidates = torch.cat([p_emb, n_emb], dim=0)   # [2B, dim]
        else:
            candidates = p_emb                              # [B, dim] -- pure in-batch negatives

        scores = torch.matmul(q_emb, candidates.t()) * self.scale
        labels = torch.arange(scores.size(0), device=scores.device)
        return self.cross_entropy(scores, labels)


@torch.no_grad()
def evaluate_retrieval_chunk_pool(model, queries: dict, corpus: dict, relevant_docs: dict,
                                   max_tokens: int = None, pooling: str = "mean",
                                   k_values=(1, 3, 5, 10), batch_size: int = 32) -> dict:
    """
    No-grad evaluation counterpart to ChunkPoolMultipleNegativesRankingLoss.
    Embeds the val/test corpus and queries with the SAME chunk-and-pool
    method used in training (rather than sentence-transformers'
    InformationRetrievalEvaluator, which would call model.encode()
    directly and silently re-truncate any long corpus paragraph).

    Computes Accuracy@k, MRR@10, and NDCG@10 -- NDCG@10 is returned as
    "primary_score", matching InformationRetrievalEvaluator's own default
    primary/best-model-selection metric, so "best epoch" is chosen the
    same way it always was, just scored on complete (not truncated) text.
    """
    model.eval()
    qids = list(queries.keys())
    cids = list(corpus.keys())

    def embed_all(ids, text_of):
        vecs = []
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = [text_of[i_] for i_ in batch_ids]
            vecs.append(encode_with_chunk_pool(model, batch_texts, max_tokens, pooling))
        return torch.cat(vecs, dim=0) if vecs else torch.zeros(0, model.get_sentence_embedding_dimension())

    q_embs = embed_all(qids, queries)
    c_embs = embed_all(cids, corpus)

    sims = torch.matmul(q_embs, c_embs.t())  # cosine, since both sides are L2-normalized

    top_k = min(max(k_values), len(cids)) if cids else 0
    top_idx = torch.topk(sims, k=top_k, dim=1).indices if top_k else torch.zeros(len(qids), 0, dtype=torch.long)

    accuracy_hits = {k: 0 for k in k_values}
    mrr_total = 0.0
    ndcg_total = 0.0
    n_eval_queries = 0

    for qi, qid in enumerate(qids):
        rel = relevant_docs.get(qid, set())
        if not rel:
            continue
        n_eval_queries += 1
        ranked_cids = [cids[j] for j in top_idx[qi].tolist()]

        for k in k_values:
            if any(cid in rel for cid in ranked_cids[:k]):
                accuracy_hits[k] += 1

        rr = 0.0
        for rank, cid in enumerate(ranked_cids, start=1):
            if cid in rel:
                rr = 1.0 / rank
                break
        mrr_total += rr

        dcg = sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(ranked_cids[:10], start=1) if cid in rel)
        ideal_hits = min(len(rel), 10)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcg_total += (dcg / idcg) if idcg > 0 else 0.0

    n_eval_queries = max(n_eval_queries, 1)
    metrics = {f"accuracy@{k}": accuracy_hits[k] / n_eval_queries for k in k_values}
    metrics["mrr@10"] = mrr_total / n_eval_queries
    metrics["ndcg@10"] = ndcg_total / n_eval_queries
    metrics["primary_score"] = metrics["ndcg@10"]
    metrics["n_eval_queries"] = n_eval_queries
    return metrics
