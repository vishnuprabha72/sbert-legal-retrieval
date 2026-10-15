"""
ablation_negatives.py -- ONE run that trains BOTH variants:
  A. in-batch negatives only  (current default)
  B. in-batch + mined hard negatives (config.yaml's negatives.hard_negative_percentile_band)

...from the SAME chunk-pool setup, the SAME set of training rows, and the
SAME batch ordering -- so the only thing that differs between the two
trained models is whether the loss ever saw the explicit hard negative.
Both are then evaluated on the same held-out test set (retrieval
accuracy/NDCG@10 via the chunk-pool-consistent evaluator, plus the
cross-source style-parity diagnostic) and a side-by-side report is
printed and saved.

WHY THE SAME ROW SET: add_hard_negatives() (stage3) can leave a handful
of rows without a mined negative (same-role cross-source pool < 3
candidates). Those rows are dropped up front, before EITHER variant
trains -- so a difference in the final numbers can't be explained by
"variant B just saw fewer/different rows", only by the negative strategy
itself.

COST: this trains two full chunk-pool models sequentially on CPU. Run
with --limit first (e.g. --limit 150) to sanity-check the whole pipeline
and get a real per-epoch timing number before committing to a full run --
the script also self-reports elapsed time for variant A and projects an
ETA for variant B before starting it.

Run:
    uv run python ablation_negatives.py --limit 150   # quick sanity/timing run first
    uv run python ablation_negatives.py                # full run
"""
import argparse
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
from stage3_build_training_table import add_hard_negatives  # noqa: E402
from stage5_train_sbert import _build_eval_sets, _train_model_chunk_pool, build_source_mixed_order  # noqa: E402


def set_all_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def style_parity_chunk_pool(model, test_df: pd.DataFrame, max_tokens=None, pooling: str = "mean") -> dict:
    """
    Chunk-pool-consistent version of stage6_evaluate.py's
    style_parity_diagnostic -- same metric (mean/std of judgment-vs-llm
    score gap for the same query), but scored with chunk-and-pool instead
    of model.encode()'s plain truncation, so long paragraphs aren't
    silently cut before the comparison.
    """
    model.eval()
    judg = test_df[test_df["positive_source"] == "judgment_original"][["query", "positive"]]
    llm = test_df[test_df["positive_source"] == "llm_generated"][["query", "positive"]]
    merged = judg.merge(llm, on="query", suffixes=("_judg", "_llm"))
    if merged.empty:
        return {"note": "No overlapping query pairs with both sources found in test set."}

    q_emb = encode_with_chunk_pool(model, merged["query"].tolist(), max_tokens, pooling).numpy()
    judg_emb = encode_with_chunk_pool(model, merged["positive_judg"].tolist(), max_tokens, pooling).numpy()
    llm_emb = encode_with_chunk_pool(model, merged["positive_llm"].tolist(), max_tokens, pooling).numpy()

    judg_scores = np.sum(q_emb * judg_emb, axis=1)
    llm_scores = np.sum(q_emb * llm_emb, axis=1)
    gap = judg_scores - llm_scores
    return {
        "n_pairs": int(len(merged)),
        "mean_gap_judgment_minus_llm": float(np.mean(gap)),
        "std_gap": float(np.std(gap)),
    }


def evaluate_variant(model_dir: str, test_df: pd.DataFrame, max_tokens, pooling: str) -> dict:
    model = SentenceTransformer(model_dir)
    queries, corpus, relevant_docs = _build_eval_sets(test_df)
    retrieval = evaluate_retrieval_chunk_pool(model, queries, corpus, relevant_docs,
                                               max_tokens=max_tokens, pooling=pooling)
    parity = style_parity_chunk_pool(model, test_df, max_tokens, pooling)
    return {"retrieval": retrieval, "style_parity": parity}


def main(cfg: dict, limit: int | None):
    stage3_dir = Path(cfg["paths"]["stage3_dir"])
    train_df = pd.read_parquet(stage3_dir / "train.parquet").dropna(subset=["query", "positive"])
    val_df = pd.read_parquet(stage3_dir / "val.parquet")
    test_df = pd.read_parquet(stage3_dir / "test.parquet").dropna(subset=["query", "positive"])

    if limit:
        train_df = train_df.head(limit).reset_index(drop=True)
        print(f"--limit {limit}: using only the first {len(train_df)} training rows (sanity/timing run)")

    # --- ensure every row has a mined hard negative, mining now if needed ---
    if "negative" not in train_df.columns or train_df["negative"].isna().any():
        print("Mining hard negatives for train_df (same-role, cross-source, similarity-rank banded)...")
        model_name = cfg["alignment"]["embedding_model"]
        band = tuple(cfg["negatives"]["hard_negative_percentile_band"])
        train_df = add_hard_negatives(train_df, model_name, band)

    before = len(train_df)
    train_df = train_df.dropna(subset=["negative"]).reset_index(drop=True)
    dropped = before - len(train_df)
    if dropped:
        print(f"Dropped {dropped}/{before} rows with no mineable hard negative -- "
              f"BOTH variants below train on the identical remaining {len(train_df)} rows.")

    batch_size = cfg["training"]["batch_size"]
    set_all_seeds(42)
    mixed_order = build_source_mixed_order(
        sources=train_df["positive_source"].tolist(),
        batch_size=batch_size,
        min_fraction=cfg["training"]["min_source_fraction_per_batch"],
    )
    ordered_df = train_df.iloc[mixed_order].reset_index(drop=True)
    print(f"Training rows (shared by both variants): {len(ordered_df)}")

    models_dir = Path(cfg["training"]["output_dir"]).parent / "ablation"
    dir_in_batch = models_dir / "in_batch"
    dir_hard_neg = models_dir / "hard_negative"

    cp_cfg = cfg["training"].get("chunk_pool", {})
    max_tokens = cp_cfg.get("max_tokens")  # None -> resolved from model.max_seq_length inside training/eval
    pooling = cp_cfg.get("pooling", "mean")

    # --- Variant A: in-batch negatives only ---
    print("\n=== Variant A: in-batch negatives only ===")
    set_all_seeds(42)
    model_a = SentenceTransformer(cfg["training"]["base_model"])
    t0 = time.time()
    _train_model_chunk_pool(ordered_df, val_df, cfg, model_a, batch_size, str(dir_in_batch),
                             verbose=True, use_hard_negatives=False)
    elapsed_a = time.time() - t0
    print(f"Variant A total time: {elapsed_a / 60:.1f} min")
    print(f"Projected time for Variant B (similar cost, plus hard-negative encoding overhead): "
          f"~{elapsed_a * 1.3 / 60:.1f}-{elapsed_a * 1.6 / 60:.1f} min")

    # --- Variant B: in-batch + mined hard negatives ---
    print("\n=== Variant B: in-batch + hard negatives ===")
    set_all_seeds(42)
    model_b = SentenceTransformer(cfg["training"]["base_model"])
    t0 = time.time()
    _train_model_chunk_pool(ordered_df, val_df, cfg, model_b, batch_size, str(dir_hard_neg),
                             verbose=True, use_hard_negatives=True)
    elapsed_b = time.time() - t0
    print(f"Variant B total time: {elapsed_b / 60:.1f} min")

    # --- Evaluate both on the same held-out test set ---
    print("\n=== Evaluating both variants on the held-out test set ===")
    report = {
        "n_train_rows": len(ordered_df),
        "in_batch": evaluate_variant(str(dir_in_batch), test_df, max_tokens, pooling),
        "hard_negative": evaluate_variant(str(dir_hard_neg), test_df, max_tokens, pooling),
        "train_time_minutes": {"in_batch": elapsed_a / 60, "hard_negative": elapsed_b / 60},
    }

    out_path = stage3_dir / "ablation_negatives_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Comparison ---")
    for k in ("ndcg@10", "mrr@10", "accuracy@1", "accuracy@10"):
        a = report["in_batch"]["retrieval"].get(k)
        b = report["hard_negative"]["retrieval"].get(k)
        print(f"  {k:12s}  in_batch={a:.4f}   hard_negative={b:.4f}   delta={b - a:+.4f}")
    pa = report["in_batch"]["style_parity"]
    pb = report["hard_negative"]["style_parity"]
    if "mean_gap_judgment_minus_llm" in pa and "mean_gap_judgment_minus_llm" in pb:
        print(f"  style_parity mean_gap   in_batch={pa['mean_gap_judgment_minus_llm']:.4f}   "
              f"hard_negative={pb['mean_gap_judgment_minus_llm']:.4f}")
        print(f"  style_parity std_gap    in_batch={pa['std_gap']:.4f}   "
              f"hard_negative={pb['std_gap']:.4f}")
    print(f"\nFull report saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                         help="Use only the first N training rows -- for a quick sanity/timing run "
                              "before committing to the full dataset.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.limit)
