"""
FIRE/AILA judgment retrieval (Section 5.2, Tables 3 and 4): per-query scores and
paired significance tests.

Input (results/fire_aila/):
    rank_detail_pool186.csv, rank_detail_pool2914.csv
        one row per (model x encoding, BM25 weight, query, query formulation,
        relevant judgment) with the rank at which that judgment was retrieved
        ("Not retrieved" if it was outside the ranking)
    summary_pool186.csv, summary_pool2914.csv
        MAP / P@10 / BPREF / MRR per configuration, as written by the
        evaluation run; used here only as a consistency check

Output (results/fire_aila/):
    per_query_pool186.csv, per_query_pool2914.csv   per-query AP and RR
    table3_encoding_vs_legacy.csv                     Table 3 (186 pool, alpha = 0)
    table4_vs_bm25.csv                                Table 4 (both pools)
    pool_size_shift.csv                               change in (semantic - BM25) AP
    hybrid_vs_dense.csv                               hybrid (alpha = 0.5) vs. dense alone

AP follows the TREC definition: precision at the rank of each retrieved relevant
judgment, summed and divided by the number of relevant judgments (relevant
judgments that were not retrieved contribute zero). All tests are paired over
the 50 AILA queries, Facts formulation: mean difference with a 95% percentile
bootstrap interval (20,000 resamples), two-sided sign-flip permutation test
(20,000 permutations) and Wilcoxon signed-rank test. W-L counts queries on
which a system's AP is higher / lower than the reference.

The 2,914-judgment pool was run with chunk-pool mean inference only.

Run from the repository root:
    uv run python evaluation/fire_aila_significance.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 20_000
N_PERM = 20_000
VARIANT = "Facts only"
KEYS = ["Model x Encoding", "Variant", "BM25 weight"]

MODELS = {
    "Legacy-trained": "Mine_Finetuned_Indian_Legal",
    "CP-trained": "Mine_Finetuned_Indian_Legal_ChunkPool",
    "MiniLM": "MiniLM (general)",
    "Swiss Legal SBERT": "Swiss Legal SBERT (domain)",
    "InLegalBERT": "InLegalBERT (Indian domain, no SBERT tuning)",
}
ENCODINGS = {"legacy": "Legacy (truncated)", "CP-mean": "Chunk-Pool Mean", "CP-max": "Chunk-Pool Max"}


# ── per-query scores ─────────────────────────────────────────────────────────

def per_query_scores(rank_file: Path) -> pd.DataFrame:
    d = pd.read_csv(rank_file)
    d["rank"] = pd.to_numeric(d["Rank"], errors="coerce")

    def score(g: pd.DataFrame) -> pd.Series:
        ranks = np.sort(g["rank"].dropna().to_numpy())
        ap = sum((i + 1) / r for i, r in enumerate(ranks)) / len(g)
        rr = 0.0 if len(ranks) == 0 else 1.0 / ranks[0]
        return pd.Series({"AP": ap, "RR": rr, "n_relevant": len(g)})

    return (d.groupby(KEYS + ["Query"]).apply(score, include_groups=False)
             .reset_index())


def check_against_summary(pq: pd.DataFrame, summary_file: Path, label: str) -> None:
    s = pd.read_csv(summary_file)
    m = pq.groupby(KEYS)[["AP", "RR"]].mean().reset_index().merge(s, on=KEYS)
    diff = max((m["AP"] - m["MAP"]).abs().max(), (m["RR"] - m["MRR"]).abs().max())
    print(f"[{label}] {len(m)} configurations; max |rebuilt - reported| = {diff:.1e}")
    if diff > 1e-4:
        raise SystemExit(f"[{label}] per-query scores do not reproduce the summary file")


def series(pq: pd.DataFrame, model: str, encoding: str, weight: float) -> pd.Series:
    sel = pq[(pq["Variant"] == VARIANT)
             & (pq["Model x Encoding"] == f"{MODELS[model]} — {ENCODINGS[encoding]}")
             & (pq["BM25 weight"] == weight)]
    return sel.set_index("Query")["AP"].sort_index()


def bm25(pq: pd.DataFrame) -> pd.Series:
    # BM25-only (weight 1.0) does not depend on the embedding model.
    sel = pq[(pq["Variant"] == VARIANT) & (pq["BM25 weight"] == 1.0)]
    return sel.groupby("Query")["AP"].first().sort_index()


# ── tests ────────────────────────────────────────────────────────────────────

def paired(a: pd.Series, b: pd.Series, rng) -> dict:
    d = (a - b).to_numpy()
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    p_perm = (np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (N_PERM + 1)
    p_wil = 1.0 if np.allclose(d, 0) else stats.wilcoxon(d).pvalue
    return {
        "delta": round(d.mean(), 4),
        "ci95_low": round(np.percentile(boot, 2.5), 4),
        "ci95_high": round(np.percentile(boot, 97.5), 4),
        "wins": int(np.sum(d > 1e-12)), "losses": int(np.sum(d < -1e-12)),
        "p_permutation": round(p_perm, 4), "p_wilcoxon": round(float(p_wil), 4),
    }


def table3(p186: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for model in ["Legacy-trained", "CP-trained", "MiniLM"]:
        legacy = series(p186, model, "legacy", 0.0)
        for enc in ["legacy", "CP-mean", "CP-max"]:
            s = series(p186, model, enc, 0.0)
            row = {"model": model, "inference": enc, "MAP": round(s.mean(), 4)}
            if enc != "legacy":
                row.update({f"vs_legacy_{k}": v for k, v in paired(s, legacy, rng).items()})
            rows.append(row)
    for enc_a, enc_b in [("CP-mean", "CP-mean"), ("legacy", "legacy")]:
        r = paired(series(p186, "Legacy-trained", enc_a, 0.0), series(p186, "CP-trained", enc_b, 0.0), rng)
        rows.append({"model": "Legacy-trained minus CP-trained", "inference": enc_a,
                     **{f"vs_legacy_{k}": v for k, v in r.items()}})
    return pd.DataFrame(rows)


def table4(pools: dict, rng) -> pd.DataFrame:
    rows = []
    systems = [(m, 0.0) for m in MODELS] + [("CP-trained", 0.5), ("Legacy-trained", 0.5)]
    for pool, pq in pools.items():
        ref = bm25(pq)
        rows.append({"pool": pool, "system": "BM25", "MAP": round(ref.mean(), 4)})
        for model, w in systems:
            s = series(pq, model, "CP-mean", w)
            name = f"Hybrid: BM25 + {model}" if w == 0.5 else model
            rows.append({"pool": pool, "system": name, "MAP": round(s.mean(), 4), **paired(s, ref, rng)})
    return pd.DataFrame(rows)


def pool_shift(pools: dict, rng) -> pd.DataFrame:
    p186, p2914 = pools["186"], pools["2914"]
    b186, b2914 = bm25(p186), bm25(p2914)
    rows = []
    for model in MODELS:
        gap186 = series(p186, model, "CP-mean", 0.0) - b186
        gap2914 = series(p2914, model, "CP-mean", 0.0) - b2914
        rows.append({"model": model, "gap_186": round(gap186.mean(), 4),
                     "gap_2914": round(gap2914.mean(), 4),
                     **{f"shift_{k}": v for k, v in paired(gap2914, gap186, rng).items()}})
    return pd.DataFrame(rows)


def hybrid_vs_dense(pools: dict, rng) -> pd.DataFrame:
    rows = []
    for pool, pq in pools.items():
        for model in ["CP-trained", "Legacy-trained"]:
            h = series(pq, model, "CP-mean", 0.5)
            d = series(pq, model, "CP-mean", 0.0)
            rows.append({"pool": pool, "model": model, "MAP_hybrid": round(h.mean(), 4),
                         "MAP_dense": round(d.mean(), 4), **paired(h, d, rng)})
    return pd.DataFrame(rows)


def bm25_ratio(pq: pd.DataFrame, model: str, rng) -> tuple[float, float, float]:
    ref = bm25(pq).to_numpy()
    sem = series(pq, model, "CP-mean", 0.0).to_numpy()
    idx = rng.integers(0, len(ref), size=(N_BOOT, len(ref)))
    boot = ref[idx].mean(axis=1) / sem[idx].mean(axis=1)
    return ref.mean() / sem.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/fire_aila")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    base = Path(args.dir)
    rng = np.random.default_rng(args.seed)

    pools = {}
    for pool in ["186", "2914"]:
        pq = per_query_scores(base / f"rank_detail_pool{pool}.csv")
        check_against_summary(pq, base / f"summary_pool{pool}.csv", f"pool {pool}")
        pq.to_csv(base / f"per_query_pool{pool}.csv", index=False)
        pools[pool] = pq

    t3 = table3(pools["186"], rng)
    t3.to_csv(base / "table3_encoding_vs_legacy.csv", index=False)
    print("\nTable 3 (186 pool, alpha = 0)\n", t3.to_string(index=False))

    t4 = table4(pools, rng)
    t4.to_csv(base / "table4_vs_bm25.csv", index=False)
    print("\nTable 4 (vs. BM25)\n", t4.to_string(index=False))

    ps = pool_shift(pools, rng)
    ps.to_csv(base / "pool_size_shift.csv", index=False)
    print("\nChange in (semantic - BM25) AP from 186 to 2,914 judgments\n", ps.to_string(index=False))

    hd = hybrid_vs_dense(pools, rng)
    hd.to_csv(base / "hybrid_vs_dense.csv", index=False)
    print("\nHybrid vs. dense alone\n", hd.to_string(index=False))

    r, lo, hi = bm25_ratio(pools["2914"], "CP-trained", rng)
    print(f"\nBM25 MAP / CP-trained MAP at 2,914 judgments: {r:.3f} (95% CI {lo:.2f}-{hi:.2f})")


if __name__ == "__main__":
    main()
