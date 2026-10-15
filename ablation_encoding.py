"""
ablation_encoding.py -- ONE run that trains THREE variants:
  A. legacy         -- model.fit(), plain truncation at max_seq_length (the original design)
  B. chunk_pool_mean -- chunk-and-pool training, mean pooling across chunks
  C. chunk_pool_max  -- chunk-and-pool training, max pooling across chunks

This is the comparison this whole chunk-pool effort was actually built to
answer, and it hadn't been run as an actual head-to-head until this script:
does chunk-and-pool training genuinely beat plain truncation, or does mean-
pooling dilute the signal the same way it was observed to in the separate
caselaw-retrieval experiment (frozen, untrained model, cross-source chunk
comparison)? That prior finding was on a FROZEN model -- it's an open
question whether fine-tuning can compensate for the dilution mean-pooling
risks, which is exactly what this ablation tests.

All three variants share the SAME training rows and the SAME batch
ordering (computed once, reused three times) and the SAME base checkpoint/
seed -- isolating "which encoding method" as the only variable, same
design principle as ablation_negatives.py's in-batch-vs-hard-negative
comparison.

EVALUATION, DELIBERATELY NOT UNIFORM: each variant is evaluated with the
SAME encoding method it was trained with (legacy -> plain model.encode()
truncation; chunk-pool variants -> chunk-and-pool evaluation) -- because
that's how each would actually be deployed. This is an "as-deployed"
comparison, not an attempt to isolate embedding quality independent of
encoding method.

COST: three full trainings, sequentially, on CPU. Run with --limit first.

MEMORY: to stay within RAM, each trained model is released as soon as it
is saved, and the variants are evaluated one at a time (load, run retrieval
and style-parity evaluation, free) rather than holding all of them in
memory.

Run:
    uv run python ablation_encoding.py --limit 150   # sanity/timing run first
    uv run python ablation_encoding.py                # full run
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")

from sentence_transformers import SentenceTransformer  # noqa: E402

from common.chunk_pool_encoding import encode_with_chunk_pool, evaluate_retrieval_chunk_pool  # noqa: E402
from common.io_utils import load_config  # noqa: E402
from stage5_train_sbert import (  # noqa: E402
    _build_eval_sets, _train_model_chunk_pool, _train_model_legacy, build_source_mixed_order,
)


def set_all_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def free_model(model):
    """Drop a model object and force memory back to the OS before the next load."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _retrieval_metrics_legacy(model: SentenceTransformer, test_df: pd.DataFrame,
                               k_values=(1, 3, 5, 10)) -> dict:
    """
    Plain-truncation evaluation counterpart to evaluate_retrieval_chunk_pool
    -- uses model.encode() directly (truncating), matching how the legacy
    variant would actually be deployed. Computes the same metric set
    (Accuracy@k, MRR@10, NDCG@10) for direct comparability with the
    chunk-pool variants' numbers. Takes an already-loaded model (no reload).
    """
    import math
    queries, corpus, relevant_docs = _build_eval_sets(test_df)
    qids, cids = list(queries.keys()), list(corpus.keys())

    q_embs = model.encode([queries[q] for q in qids], normalize_embeddings=True, show_progress_bar=False)
    c_embs = model.encode([corpus[c] for c in cids], normalize_embeddings=True, show_progress_bar=False)
    sims = q_embs @ c_embs.T

    top_k = min(max(k_values), len(cids))
    top_idx = np.argsort(-sims, axis=1)[:, :top_k]

    accuracy_hits = {k: 0 for k in k_values}
    mrr_total = 0.0
    ndcg_total = 0.0
    n_eval = 0
    for qi, qid in enumerate(qids):
        rel = relevant_docs.get(qid, set())
        if not rel:
            continue
        n_eval += 1
        ranked = [cids[j] for j in top_idx[qi]]
        for k in k_values:
            if any(cid in rel for cid in ranked[:k]):
                accuracy_hits[k] += 1
        rr = 0.0
        for rank, cid in enumerate(ranked, start=1):
            if cid in rel:
                rr = 1.0 / rank
                break
        mrr_total += rr
        dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(ranked[:10], start=1) if cid in rel)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(rel), 10) + 1))
        ndcg_total += (dcg / idcg) if idcg > 0 else 0.0

    n_eval = max(n_eval, 1)
    metrics = {f"accuracy@{k}": accuracy_hits[k] / n_eval for k in k_values}
    metrics["mrr@10"] = mrr_total / n_eval
    metrics["ndcg@10"] = ndcg_total / n_eval
    metrics["primary_score"] = metrics["ndcg@10"]
    return metrics


@torch.no_grad()
def _style_parity_legacy(model: SentenceTransformer, test_df: pd.DataFrame) -> dict:
    """Same as before, but takes an already-loaded model (no reload)."""
    judg = test_df[test_df["positive_source"] == "judgment_original"][["query", "positive"]]
    llm = test_df[test_df["positive_source"] == "llm_generated"][["query", "positive"]]
    merged = judg.merge(llm, on="query", suffixes=("_judg", "_llm"))
    if merged.empty:
        return {"note": "No overlapping query pairs with both sources found in test set."}
    q_emb = model.encode(merged["query"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    judg_emb = model.encode(merged["positive_judg"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    llm_emb = model.encode(merged["positive_llm"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    gap = np.sum(q_emb * judg_emb, axis=1) - np.sum(q_emb * llm_emb, axis=1)
    return {"n_pairs": int(len(merged)), "mean_gap_judgment_minus_llm": float(np.mean(gap)),
            "std_gap": float(np.std(gap))}


@torch.no_grad()
def style_parity_chunk_pool(model, test_df: pd.DataFrame, max_tokens=None, pooling: str = "mean") -> dict:
    model.eval()
    judg = test_df[test_df["positive_source"] == "judgment_original"][["query", "positive"]]
    llm = test_df[test_df["positive_source"] == "llm_generated"][["query", "positive"]]
    merged = judg.merge(llm, on="query", suffixes=("_judg", "_llm"))
    if merged.empty:
        return {"note": "No overlapping query pairs with both sources found in test set."}
    q_emb = encode_with_chunk_pool(model, merged["query"].tolist(), max_tokens, pooling).numpy()
    judg_emb = encode_with_chunk_pool(model, merged["positive_judg"].tolist(), max_tokens, pooling).numpy()
    llm_emb = encode_with_chunk_pool(model, merged["positive_llm"].tolist(), max_tokens, pooling).numpy()
    gap = np.sum(q_emb * judg_emb, axis=1) - np.sum(q_emb * llm_emb, axis=1)
    return {"n_pairs": int(len(merged)), "mean_gap_judgment_minus_llm": float(np.mean(gap)),
            "std_gap": float(np.std(gap))}


def evaluate_legacy_variant_full(model_dir: str, test_df: pd.DataFrame) -> dict:
    """Load the legacy model ONCE, run both retrieval + style-parity off it, then free it."""
    print(f"  Loading {model_dir} ...")
    model = SentenceTransformer(model_dir)
    result = {
        "retrieval": _retrieval_metrics_legacy(model, test_df),
        "style_parity": _style_parity_legacy(model, test_df),
    }
    free_model(model)
    return result


def evaluate_chunk_pool_variant_full(model_dir: str, test_df: pd.DataFrame, max_tokens, pooling: str) -> dict:
    """Load the chunk-pool model ONCE, run both retrieval + style-parity off it, then free it."""
    print(f"  Loading {model_dir} ...")
    model = SentenceTransformer(model_dir)
    queries, corpus, relevant_docs = _build_eval_sets(test_df)
    retrieval = evaluate_retrieval_chunk_pool(model, queries, corpus, relevant_docs,
                                               max_tokens=max_tokens, pooling=pooling)
    parity = style_parity_chunk_pool(model, test_df, max_tokens, pooling)
    free_model(model)
    return {"retrieval": retrieval, "style_parity": parity}


def main(cfg: dict, limit: int | None):
    stage3_dir = Path(cfg["paths"]["stage3_dir"])
    train_df = pd.read_parquet(stage3_dir / "train.parquet").dropna(subset=["query", "positive"])
    val_df = pd.read_parquet(stage3_dir / "val.parquet")
    test_df = pd.read_parquet(stage3_dir / "test.parquet").dropna(subset=["query", "positive"])

    if limit:
        train_df = train_df.head(limit).reset_index(drop=True)
        print(f"--limit {limit}: using only the first {len(train_df)} training rows (sanity/timing run)")

    batch_size = cfg["training"]["batch_size"]
    set_all_seeds(42)
    mixed_order = build_source_mixed_order(
        sources=train_df["positive_source"].tolist(), batch_size=batch_size,
        min_fraction=cfg["training"]["min_source_fraction_per_batch"])
    ordered_df = train_df.iloc[mixed_order].reset_index(drop=True)
    print(f"Training rows (shared by all 3 variants): {len(ordered_df)}")

    models_dir = Path(cfg["training"]["output_dir"]).parent / "ablation_encoding"
    dirs = {
        "legacy": models_dir / "legacy",
        "chunk_pool_mean": models_dir / "chunk_pool_mean",
        "chunk_pool_max": models_dir / "chunk_pool_max",
    }
    cp_cfg = cfg["training"].get("chunk_pool", {})
    max_tokens = cp_cfg.get("max_tokens")
    timings = {}

    print("\n=== Variant A: legacy (plain truncation) ===")
    set_all_seeds(42)
    model_a = SentenceTransformer(cfg["training"]["base_model"])
    t0 = time.time()
    _train_model_legacy(ordered_df, val_df, cfg, model_a, batch_size, str(dirs["legacy"]), verbose=True)
    timings["legacy"] = time.time() - t0
    print(f"Variant A total time: {timings['legacy'] / 60:.1f} min")
    free_model(model_a)  # already saved to disk by _train_model_legacy -- don't need it in RAM anymore

    print("\n=== Variant B: chunk-pool (mean pooling) ===")
    set_all_seeds(42)
    model_b = SentenceTransformer(cfg["training"]["base_model"])
    t0 = time.time()
    _train_model_chunk_pool(ordered_df, val_df, cfg, model_b, batch_size, str(dirs["chunk_pool_mean"]),
                             verbose=True, pooling="mean")
    timings["chunk_pool_mean"] = time.time() - t0
    print(f"Variant B total time: {timings['chunk_pool_mean'] / 60:.1f} min "
          f"(projected Variant C: ~{timings['chunk_pool_mean'] / 60:.1f} min, same encoding cost, "
          f"pooling method doesn't change chunk count)")
    free_model(model_b)

    print("\n=== Variant C: chunk-pool (max pooling) ===")
    set_all_seeds(42)
    model_c = SentenceTransformer(cfg["training"]["base_model"])
    t0 = time.time()
    _train_model_chunk_pool(ordered_df, val_df, cfg, model_c, batch_size, str(dirs["chunk_pool_max"]),
                             verbose=True, pooling="max")
    timings["chunk_pool_max"] = time.time() - t0
    print(f"Variant C total time: {timings['chunk_pool_max'] / 60:.1f} min")
    free_model(model_c)

    print("\n=== Evaluating all 3 variants on the held-out test set (one at a time, to keep peak RAM low) ===")
    report = {
        "n_train_rows": len(ordered_df),
        "train_time_minutes": {k: v / 60 for k, v in timings.items()},
    }

    print("\n-- Evaluating: legacy --")
    report["legacy"] = evaluate_legacy_variant_full(str(dirs["legacy"]), test_df)

    print("\n-- Evaluating: chunk_pool_mean --")
    report["chunk_pool_mean"] = evaluate_chunk_pool_variant_full(
        str(dirs["chunk_pool_mean"]), test_df, max_tokens, "mean")

    print("\n-- Evaluating: chunk_pool_max --")
    report["chunk_pool_max"] = evaluate_chunk_pool_variant_full(
        str(dirs["chunk_pool_max"]), test_df, max_tokens, "max")

    out_path = stage3_dir / "ablation_encoding_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Comparison ---")
    variants = {
        "legacy": report["legacy"]["retrieval"],
        "chunk_pool_mean": report["chunk_pool_mean"]["retrieval"],
        "chunk_pool_max": report["chunk_pool_max"]["retrieval"],
    }
    for k in ("ndcg@10", "mrr@10", "accuracy@1", "accuracy@10"):
        vals = {name: variants[name].get(k) for name in variants}
        print(f"  {k:12s} " + "  ".join(f"{name}={v:.4f}" for name, v in vals.items()))
    best = max(variants, key=lambda name: variants[name]["ndcg@10"])
    print(f"\nBest by NDCG@10 (primary metric): {best}")
    print(f"Full report saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                         help="Use only the first N training rows -- for a quick sanity/timing run "
                              "before committing to the full dataset.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.limit)
