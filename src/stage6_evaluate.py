"""
Stage 6 — Evaluate the fine-tuned model on two dimensions:

  A. Standard retrieval accuracy (top-k / MRR) per role and per query_style
  B. Cross-source style-parity diagnostic: for queries whose correct answer
     exists as BOTH a judgment_original and llm_generated positive, check
     whether the model ranks/scores both sources comparably (i.e. it isn't
     favoring one source's surface style over genuine semantic match).

Input:  data/models/sbert-legal-v1/ (fine-tuned model)
        data/stage3_training_table/test.parquet
Output: prints a report; also saves data/stage3_training_table/eval_report.json

Run:
    uv run python src/stage6_evaluate.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from common.io_utils import load_config


def top_k_accuracy(model, df: pd.DataFrame, k: int = 5) -> dict:
    """For each query, build a candidate pool (all unique positives in the
    test set) and check whether its true positive is ranked in the top-k."""
    candidates = df["positive"].unique().tolist()
    cand_idx = {c: i for i, c in enumerate(candidates)}
    cand_embs = model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)

    results_by_role = {}
    results_by_style = {}

    queries = df["query"].tolist()
    query_embs = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)

    hits = 0
    mrr_total = 0.0
    for i, row in df.reset_index(drop=True).iterrows():
        sims = cand_embs @ query_embs[i]
        ranked = np.argsort(-sims)
        true_idx = cand_idx[row["positive"]]
        rank = int(np.where(ranked == true_idx)[0][0]) + 1
        hit = rank <= k
        hits += hit
        mrr_total += 1.0 / rank

        results_by_role.setdefault(row["role"], []).append(hit)
        results_by_style.setdefault(row["query_style"], []).append(hit)

    n = len(df)
    return {
        "top_k": k,
        "overall_accuracy": hits / n if n else 0,
        "mrr": mrr_total / n if n else 0,
        "by_role": {r: float(np.mean(v)) for r, v in results_by_role.items()},
        "by_query_style": {s: float(np.mean(v)) for s, v in results_by_style.items()},
    }


def style_parity_diagnostic(model, df: pd.DataFrame) -> dict:
    """
    For queries that have BOTH a judgment_original and llm_generated positive
    for the same underlying content (i.e. same query + role, both source rows
    exist), compare similarity scores across source. A large systematic gap
    indicates style bias.
    """
    judg = df[df["positive_source"] == "judgment_original"][["query", "positive", "role"]]
    llm = df[df["positive_source"] == "llm_generated"][["query", "positive", "role"]]
    merged = judg.merge(llm, on="query", suffixes=("_judg", "_llm"))

    if merged.empty:
        return {"note": "No overlapping query pairs with both sources found in test set."}

    q_embs = model.encode(merged["query"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
    judg_embs = model.encode(merged["positive_judg"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
    llm_embs = model.encode(merged["positive_llm"].tolist(), convert_to_numpy=True, normalize_embeddings=True)

    judg_scores = np.sum(q_embs * judg_embs, axis=1)
    llm_scores = np.sum(q_embs * llm_embs, axis=1)
    gap = judg_scores - llm_scores

    return {
        "n_pairs": len(merged),
        "mean_judgment_score": float(np.mean(judg_scores)),
        "mean_llm_score": float(np.mean(llm_scores)),
        "mean_gap_judgment_minus_llm": float(np.mean(gap)),
        "std_gap": float(np.std(gap)),
        "interpretation": (
            "Gap close to 0 => good style invariance. "
            "Large positive gap => model favors judgment-paragraph style. "
            "Large negative gap => model favors LLM-summary style."
        ),
    }


def main(cfg: dict, model_path: str = None, out_path: str = None):
    model_path = model_path or cfg["training"]["output_dir"]
    stage3_dir = Path(cfg["paths"]["stage3_dir"])
    test_df = pd.read_parquet(stage3_dir / "test.parquet").dropna(subset=["query", "positive"])

    print(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)

    print("Running top-k retrieval accuracy...")
    retrieval_report = top_k_accuracy(model, test_df, k=5)

    print("Running cross-source style-parity diagnostic...")
    parity_report = style_parity_diagnostic(model, test_df)

    report = {"model_path": model_path, "retrieval_accuracy": retrieval_report, "style_parity": parity_report}

    # Default output path is derived from the model folder name, not always
    # the same eval_report.json -- otherwise evaluating multiple size_XXXX
    # models one after another would silently overwrite each other's report.
    if out_path is None:
        model_name = Path(model_path).name  # e.g. "size_0500" or "sbert-legal-v1"
        out_path = stage3_dir / f"eval_report_{model_name}.json"
    else:
        out_path = Path(out_path)

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved report -> {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-path", default=None,
                         help="Path to a specific model folder to evaluate, e.g. "
                              "data/models/learning_curve/size_0500 (default: config's training.output_dir)")
    parser.add_argument("--output", default=None,
                         help="Where to save the report JSON (default: auto-named from the model folder, "
                              "e.g. eval_report_size_0500.json, so evaluating multiple models doesn't overwrite)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, model_path=args.model_path, out_path=args.output)
