"""
Paired significance tests for the controlled paragraph-retrieval evaluation
(Section 5.1, Tables 1 and 2).

Input:  results/controlled_eval/per_query_results.csv
        data/controlled_eval/controlled_evaluation_query_set.csv  (query families)
Output: results/controlled_eval/significance_vs_bm25.csv      (Table 2 columns)
        results/controlled_eval/significance_pairwise.csv     (comparisons in the text)
        results/controlled_eval/overlap_trend.csv             (RQ2 trend test, all configs)

Methods (all paired over the 120 queries):
  - Hit@10: exact McNemar test (two-sided binomial test on discordant pairs).
  - Reciprocal rank: mean difference with a 95% percentile bootstrap interval
    (20,000 resamples), a two-sided sign-flip permutation test (20,000
    permutations) and a Wilcoxon signed-rank test. Holm correction is applied
    over the ten rows of Table 2.
  - Robustness: the same bootstrap resampling the 24 query families (queries
    sharing a gold paragraph) instead of individual queries.
  - RQ2 trend: logistic regression of "semantic wins" (vs. BM25 win or tie) on
    the ordered overlap bucket (0, 1-2, 3+), likelihood-ratio test.

Resampling uses a fixed seed, so results are reproducible; p-values and
intervals can differ from the paper in the last digit depending on the seed.

Run from the repository root:
    uv run python evaluation/significance_tests.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

N_BOOT = 20_000
N_PERM = 20_000

LABELS = {
    "Mine_Finetuned_Indian_Legal": "Legacy-trained",
    "Mine_Finetuned_Indian_Legal_ChunkPool": "CP-trained",
    "MiniLM (general)": "MiniLM",
    "InLegalBERT (Indian domain, no SBERT tuning)": "InLegalBERT",
    "Swiss Legal SBERT (domain)": "Swiss Legal SBERT",
}
ENC = {"Legacy (truncated)": "legacy", "Chunk-Pool Mean": "CP-mean"}


# ── tests ────────────────────────────────────────────────────────────────────

def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """a, b: boolean hit vectors. Returns (a-only, b-only, two-sided p)."""
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n = a_only + b_only
    p = 1.0 if n == 0 else stats.binomtest(a_only, n, 0.5).pvalue
    return a_only, b_only, p


def bootstrap_ci(diff: np.ndarray, rng, groups: np.ndarray | None = None) -> tuple[float, float]:
    """95% percentile CI of the mean difference; resamples groups if given."""
    if groups is None:
        idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
        means = diff[idx].mean(axis=1)
    else:
        uniq = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in uniq]
        means = np.empty(N_BOOT)
        for i in range(N_BOOT):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            means[i] = np.concatenate([diff[members[j]] for j in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def permutation_p(diff: np.ndarray, rng) -> float:
    """Two-sided paired sign-flip permutation test on the mean difference."""
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(diff)))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1))


def wilcoxon_p(diff: np.ndarray) -> float:
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def logistic_trend_p(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Likelihood-ratio test for slope in logit(P(y=1)) = b0 + b1*x."""
    def nll(beta):
        z = beta[0] + beta[1] * x
        return float(np.sum(np.logaddexp(0, z) - y * z))
    full = optimize.minimize(nll, np.zeros(2), method="BFGS")
    p0 = np.clip(y.mean(), 1e-9, 1 - 1e-9)
    null_nll = -float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
    lr = 2 * (null_nll - full.fun)
    return float(full.x[1]), float(stats.chi2.sf(lr, 1))


# ── analysis ─────────────────────────────────────────────────────────────────

def load(per_query: str, query_set: str) -> pd.DataFrame:
    df = pd.read_csv(per_query)
    fam = pd.read_csv(query_set)[["query_id", "query_family_id"]]
    df = df.merge(fam, on="query_id", how="left")
    df["config"] = df["model"].map(LABELS) + " (" + df["encoding"].map(ENC) + ")"
    df["hit10_sem"] = df["rank_faiss_only"] <= 10
    df["hit10_bm25"] = df["rank_bm25_only"] <= 10
    df["rr_sem"] = 1.0 / df["rank_faiss_only"]
    df["rr_bm25"] = 1.0 / df["rank_bm25_only"]
    return df.sort_values(["config", "query_id"])


def vs_bm25(df: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for config, g in df.groupby("config"):
        diff = (g["rr_sem"] - g["rr_bm25"]).to_numpy()
        a_only, b_only, p_mcn = mcnemar_exact(g["hit10_sem"].to_numpy(), g["hit10_bm25"].to_numpy())
        lo, hi = bootstrap_ci(diff, rng)
        clo, chi = bootstrap_ci(diff, rng, g["query_family_id"].to_numpy())
        rows.append({
            "config": config,
            "hit10": round(g["hit10_sem"].mean(), 4),
            "hit10_bm25": round(g["hit10_bm25"].mean(), 4),
            "discordant_sem_only": a_only, "discordant_bm25_only": b_only,
            "p_mcnemar": round(p_mcn, 4),
            "mrr": round(g["rr_sem"].mean(), 4),
            "mrr_bm25": round(g["rr_bm25"].mean(), 4),
            "delta_mrr": round(diff.mean(), 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "p_permutation": round(permutation_p(diff, rng), 4),
            "p_wilcoxon": round(wilcoxon_p(diff), 4),
            "cluster_ci95_low": round(clo, 4), "cluster_ci95_high": round(chi, 4),
        })
    out = pd.DataFrame(rows).sort_values("hit10", ascending=False)
    out["p_permutation_holm"] = [round(p, 4) for p in holm(out["p_permutation"].tolist())]
    out["significant_after_holm"] = out["p_permutation_holm"] < 0.05
    return out


def pairwise(df: pd.DataFrame, pairs: list[tuple[str, str]], rng) -> pd.DataFrame:
    wide_rr = df.pivot(index="query_id", columns="config", values="rr_sem")
    wide_hit = df.pivot(index="query_id", columns="config", values="hit10_sem")
    rows = []
    for a, b in pairs:
        diff = (wide_rr[a] - wide_rr[b]).to_numpy()
        a_only, b_only, p_mcn = mcnemar_exact(wide_hit[a].to_numpy(bool), wide_hit[b].to_numpy(bool))
        lo, hi = bootstrap_ci(diff, rng)
        rows.append({
            "comparison": f"{a} minus {b}",
            "delta_mrr": round(diff.mean(), 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "p_permutation": round(permutation_p(diff, rng), 4),
            "p_wilcoxon": round(wilcoxon_p(diff), 4),
            "hit10_discordant": f"{a_only} vs {b_only}", "p_mcnemar": round(p_mcn, 4),
        })
    return pd.DataFrame(rows)


def overlap_trend(df: pd.DataFrame) -> pd.DataFrame:
    n = df["bm25_n_matched_terms_at_gold"]
    df = df.assign(bucket=np.select([n == 0, n <= 2], [0, 1], 2),
                   sem_win=(df["winner_(lower_rank)"] == "faiss").astype(float))
    rows = []
    for config, g in df.groupby("config"):
        slope, p = logistic_trend_p(g["sem_win"].to_numpy(), g["bucket"].to_numpy(float))
        rows.append({"config": config, "slope_per_bucket": round(slope, 3), "p_trend": p})
    return pd.DataFrame(rows).sort_values("p_trend")


PAIRS = [
    ("Legacy-trained (legacy)", "MiniLM (legacy)"),
    ("Legacy-trained (legacy)", "InLegalBERT (legacy)"),
    ("Legacy-trained (legacy)", "Swiss Legal SBERT (legacy)"),
    ("Legacy-trained (legacy)", "Legacy-trained (CP-mean)"),
    ("MiniLM (legacy)", "MiniLM (CP-mean)"),
    ("CP-trained (legacy)", "CP-trained (CP-mean)"),
    ("Legacy-trained (legacy)", "CP-trained (legacy)"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/controlled_eval/per_query_results.csv")
    parser.add_argument("--query-set", default="data/controlled_eval/controlled_evaluation_query_set.csv")
    parser.add_argument("--out-dir", default="results/controlled_eval")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    df = load(args.input, args.query_set)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t2 = vs_bm25(df, np.random.default_rng(args.seed))
    t2.to_csv(out / "significance_vs_bm25.csv", index=False)
    print("Table 2: each configuration vs. BM25\n",
          t2[["config", "hit10", "p_mcnemar", "mrr", "delta_mrr", "ci95_low", "ci95_high",
              "p_permutation", "significant_after_holm"]].to_string(index=False), "\n")

    pw = pairwise(df, PAIRS, np.random.default_rng(args.seed + 1))
    pw.to_csv(out / "significance_pairwise.csv", index=False)
    print("Pairwise comparisons\n", pw.to_string(index=False), "\n")

    tr = overlap_trend(df)
    tr.to_csv(out / "overlap_trend.csv", index=False)
    print("RQ2 overlap trend (logistic, semantic win ~ bucket)\n", tr.to_string(index=False))


if __name__ == "__main__":
    main()
