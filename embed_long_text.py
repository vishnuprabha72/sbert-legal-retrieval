"""
Embed a long real-world paragraph correctly -- using chunk-and-pool instead
of naive model.encode(), which silently truncates anything beyond the
model's max_seq_length (256 tokens for all-MiniLM-L6-v2) and drops the rest
with no warning.

Also useful for comparing naive-truncated vs. chunk-and-pooled retrieval
ranking on a specific long paragraph, to see the practical difference.

Run:
    uv run python embed_long_text.py --model-path data/models/sbert-legal-v1 \
        --text "very long paragraph text here..."
    uv run python embed_long_text.py --model-path data/models/sbert-legal-v1 \
        --text-file long_paragraph.txt --query "grounds for eviction"
"""
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, "src")
from common.text_utils import chunk_and_pool_embed_by_tokens  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text", default=None, help="The long text to embed")
    parser.add_argument("--text-file", default=None, help="Path to a file containing the long text instead")
    parser.add_argument("--query", default=None,
                         help="Optional: also embed this query and show its similarity to the "
                              "long text, comparing naive-truncated vs. chunk-and-pooled")
    parser.add_argument("--pooling", default="mean", choices=["mean", "max"])
    args = parser.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Provide either --text or --text-file")

    print(f"Loading model from {args.model_path}")
    model = SentenceTransformer(args.model_path)

    n_words = len(text.split())
    n_tokens = len(model.tokenizer.encode(text, add_special_tokens=False))
    print(f"\nInput: {n_words} words, {n_tokens} tokens (model's max_seq_length = {model.max_seq_length})")

    if n_tokens > model.max_seq_length:
        pct_lost = 100 * (n_tokens - model.max_seq_length) / n_tokens
        print(f"WARNING: naive model.encode() would silently truncate this, losing the last "
              f"~{pct_lost:.0f}% of the content with no error or warning.")

    naive_emb = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    pooled_emb = chunk_and_pool_embed_by_tokens(text, model, pooling=args.pooling)

    agreement = float(naive_emb @ pooled_emb)
    print(f"\nCosine similarity between naive-truncated and chunk-and-pooled embeddings: {agreement:.4f}")
    print("(Lower values here mean the truncated-away content was substantively different from the "
          "opening -- i.e. naive encoding is losing meaningful signal, not just redundant repetition.)")

    if args.query:
        query_emb = model.encode(args.query, normalize_embeddings=True, show_progress_bar=False)
        naive_score = float(naive_emb @ query_emb)
        pooled_score = float(pooled_emb @ query_emb)
        print(f"\nQuery: {args.query!r}")
        print(f"  Similarity using naive (truncated) embedding:      {naive_score:.4f}")
        print(f"  Similarity using chunk-and-pooled (full) embedding: {pooled_score:.4f}")
        if abs(naive_score - pooled_score) > 0.05:
            print("  -> Meaningful difference: whether you chunk or not could change retrieval ranking here.")


if __name__ == "__main__":
    main()
