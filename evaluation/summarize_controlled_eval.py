"""
Summarise the Section 5.1 controlled evaluation from its per-query results.

Input:  results/controlled_eval/per_query_results.csv
        (the per-query CSV downloaded from controlled_eval_app.py's batch mode:
        120 queries x 5 models x 2 encodings = 1,200 rows)
Output: results/controlled_eval/win_rate_by_overlap.csv   (Table 1)
        results/controlled_eval/bm25_ahead_by_config.csv   (Table 1, last column)
        results/controlled_eval/per_style_summary.csv     (per-query-type breakdown, Section 5.1 text)

Table 1 compares, per query, the rank BM25-only gives the gold paragraph with the
rank given by one semantic configuration (default: Legacy-trained, legacy
encoding). Queries are bucketed by how many distinct preprocessed query terms
occur in the gold paragraph (0, 1-2, 3+). Win shares carry 95% Wilson intervals;
p is an exact two-sided sign test with ties excluded. The last column counts, for
each bucket, in how many of the ten model/encoding configurations BM25 wins more
queries than the semantic model.

The per-style table averages Hit@1 and MRR over all ten semantic
configurations; BM25's figures do not depend on the model.

Run from the repository root:
    uv run python evaluation/summarize_controlled_eval.py
"""
import argparse
import math
from pathlib import Path

import pandas as pd


BUCKETS = ["Zero (0)", "Low (1-2)", "High (3+)"]


def _bucket(n_matched: pd.Series) -> pd.Series:
    return pd.cut(n_matched, [-1, 0, 2, float("inf")], labels=BUCKETS)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def sign_test(a: int, b: int) -> float:
    """Exact two-sided sign test on a wins vs. b wins (ties excluded)."""
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def win_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(overlap=_bucket(df["bm25_n_matched_terms_at_gold"]))
    return (pd.crosstab(df["overlap"], df["winner_(lower_rank)"])
            .reindex(index=BUCKETS, columns=["bm25", "faiss", "tie"], fill_value=0))


def win_rate_by_overlap(df: pd.DataFrame, model: str, encoding: str) -> pd.DataFrame:
    sub = df[(df["model"] == model) & (df["encoding"] == encoding)]
    if sub.empty:
        raise SystemExit(f"No rows for model={model!r}, encoding={encoding!r}")
    counts = win_counts(sub)
    rows = []
    for bucket, c in counts.iterrows():
        n = int(c.sum())
        row = {"overlap": bucket, "N": n}
        for name, k in [("bm25", c["bm25"]), ("semantic", c["faiss"])]:
            lo, hi = wilson(int(k), n)
            row[f"{name}_wins_pct"] = round(100 * k / n, 1)
            row[f"{name}_ci95"] = f"[{100 * lo:.1f}, {100 * hi:.1f}]"
        row["tie_pct"] = round(100 * c["tie"] / n, 1)
        row["sign_test_p"] = round(sign_test(int(c["bm25"]), int(c["faiss"])), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def bm25_ahead_by_config(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, encoding), sub in df.groupby(["model", "encoding"]):
        counts = win_counts(sub)
        for bucket, c in counts.iterrows():
            rows.append({"model": model, "encoding": encoding, "overlap": bucket,
                         "bm25_wins": int(c["bm25"]), "semantic_wins": int(c["faiss"]),
                         "ties": int(c["tie"]), "bm25_ahead": bool(c["bm25"] > c["faiss"])})
    return pd.DataFrame(rows)


def per_style_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("query_style").agg(
        N=("query_id", "nunique"),
        semantic_hit1=("hit@1_faiss", "mean"),
        semantic_mrr=("rr_faiss", "mean"),
        bm25_hit1=("hit@1_bm25", "mean"),
        bm25_mrr=("rr_bm25", "mean"),
    ).round(4)
    out["leader_hit1"] = (out["bm25_hit1"] > out["semantic_hit1"]).map({True: "bm25", False: "semantic"})
    out["leader_mrr"] = (out["bm25_mrr"] > out["semantic_mrr"]).map({True: "bm25", False: "semantic"})
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/controlled_eval/per_query_results.csv")
    parser.add_argument("--out-dir", default="results/controlled_eval")
    parser.add_argument("--model", default="Mine_Finetuned_Indian_Legal",
                        help="semantic model for Table 1 (default: Legacy-trained)")
    parser.add_argument("--encoding", default="Legacy (truncated)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = win_rate_by_overlap(df, args.model, args.encoding)
    ahead = bm25_ahead_by_config(df)
    n_cfg = ahead.groupby(["model", "encoding"]).ngroups
    ahead_count = ahead.groupby("overlap", observed=True)["bm25_ahead"].sum().reindex(BUCKETS)
    t1["bm25_ahead_configs"] = [f"{int(ahead_count[b])}/{n_cfg}" for b in t1["overlap"]]
    t1.to_csv(out_dir / "win_rate_by_overlap.csv", index=False)
    ahead.to_csv(out_dir / "bm25_ahead_by_config.csv", index=False)
    print(f"Table 1: BM25 vs {args.model} ({args.encoding})\n", t1.to_string(index=False), "\n")

    styles = per_style_summary(df)
    styles.to_csv(out_dir / "per_style_summary.csv", index=False)
    print("Per-query-type summary (semantic = mean over all model/encoding configurations)\n",
          styles.to_string(index=False))


if __name__ == "__main__":
    main()
