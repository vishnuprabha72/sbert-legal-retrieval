"""
Explore real held-out test-set queries against the fine-tuned model --
no more hand-copying paragraph text into a script.

Two things this shows, per sampled query:
  1. Retrieval rank: where does the TRUE positive land among a pool of
     distractor paragraphs (all unique positives in the test set)?
  2. Source-invariance check: this query's judgment_original positive and
     llm_generated positive (same underlying content, two different
     "styles") -- do they score similarly? This is the core design goal
     of the whole dual-positive training setup.

Run from the project root:
    uv run python explore_test_set.py
    uv run python explore_test_set.py --n 10 --seed 42
    uv run python explore_test_set.py --role LAW      # only sample queries from one role
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "src")
from common.io_utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n", type=int, default=5, help="Number of queries to sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--role", default=None, help="Only sample queries with this role (e.g. LAW, FACT)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for the retrieval-rank check")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage3_dir = Path(cfg["paths"]["stage3_dir"])
    model_dir = cfg["training"]["output_dir"]

    df = pd.read_parquet(stage3_dir / "test.parquet")
    if args.role:
        df = df[df["role"] == args.role]
        if df.empty:
            raise SystemExit(f"No test rows found for role={args.role!r}. "
                              f"Available roles: {sorted(pd.read_parquet(stage3_dir / 'test.parquet')['role'].unique())}")

    print(f"Loading model from: {model_dir}")
    model = SentenceTransformer(model_dir)

    # Full candidate pool for the retrieval-rank check: every unique
    # positive text in the test set (same approach stage6_evaluate.py uses).
    candidates = df["positive"].unique().tolist()
    cand_embs = model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    cand_idx = {c: i for i, c in enumerate(candidates)}

    # Sample distinct queries (a query appears twice in the table -- once
    # per positive_source -- so dedupe before sampling).
    unique_queries = df.drop_duplicates(subset=["query", "judgment_id", "role"])
    sample = unique_queries.sample(n=min(args.n, len(unique_queries)), random_state=args.seed)

    for _, row in sample.iterrows():
        query = row["query"]
        role = row["role"]
        jid = row["judgment_id"]

        print("\n" + "=" * 100)
        print(f"QUERY [{role} | judgment {jid} | style={row['query_style']}]: {query}")

        # --- 1. Retrieval rank of the true positive ---
        query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        sims = cand_embs @ query_emb
        ranked = np.argsort(-sims)
        true_idx = cand_idx[row["positive"]]
        rank = int(np.where(ranked == true_idx)[0][0]) + 1
        hit = "✓ IN TOP-K" if rank <= args.top_k else "✗ missed top-k"
        print(f"  Retrieval rank of true positive: #{rank} out of {len(candidates)} candidates  [{hit}]")

        # --- 2. Source-invariance: find BOTH positives for this exact query ---
        pair = df[(df["query"] == query) & (df["judgment_id"] == jid) & (df["role"] == role)]
        by_source = {r["positive_source"]: r["positive"] for _, r in pair.iterrows()}
        if "judgment_original" in by_source and "llm_generated" in by_source:
            texts = [by_source["judgment_original"], by_source["llm_generated"]]
            embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            score_orig = float(embs[0] @ query_emb)
            score_llm = float(embs[1] @ query_emb)
            gap = score_orig - score_llm
            print(f"  Style-invariance check:")
            print(f"    judgment_original score: {score_orig:.4f}  | {by_source['judgment_original'][:80]}")
            print(f"    llm_generated    score: {score_llm:.4f}  | {by_source['llm_generated'][:80]}")
            print(f"    gap (orig - llm): {gap:+.4f}  "
                  f"{'(close to 0 = good style invariance)' if abs(gap) < 0.1 else '(noticeable gap)'}")
        else:
            print("  (only one source present for this query in the test set -- no pair to compare)")


if __name__ == "__main__":
    main()
