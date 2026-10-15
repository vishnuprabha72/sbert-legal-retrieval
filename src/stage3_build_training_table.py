"""
Stage 3 — Assemble the final training table.

For every valid (query_style, query_text) from Stage 2, emit TWO positive rows:
  - (query, judgment_paragraph, positive_source=judgment_original)
  - (query, llm_summary_sentence, positive_source=llm_generated)

If negatives.strategy == "hard_negative" in config, also mine a same-role
cross-source hard negative per row using similarity-rank banding (no clustering).
Default is "in_batch" — negatives are left blank and MultipleNegativesRankingLoss
uses in-batch negatives at training time instead.

Input:  data/stage2_queries/{id}_queries.json
Output: data/stage3_training_table/training_table.parquet

Run:
    uv run python src/stage3_build_training_table.py --limit 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common.io_utils import load_config, load_json, iter_judgment_ids
from common.embedding_utils import embed, cosine_sim_matrix


def build_rows(judgment_id: str, records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        role = rec["summary_role"]  # stage1 dropped the redundant 'role' alias; summary_role is the source of truth now
        judgment_paragraph = rec["segment_text"]
        llm_summary_sentence = rec["summary_sentence"]
        alignment_score = rec["similarity_score"]

        for style, info in rec["queries"].items():
            if not info["valid"]:
                continue
            query = info["text"]

            rows.append({
                "query": query, "query_style": style, "query_source": "llm_generated",
                "positive": judgment_paragraph, "positive_source": "judgment_original",
                "role": role, "judgment_id": judgment_id, "alignment_score": alignment_score,
            })
            rows.append({
                "query": query, "query_style": style, "query_source": "llm_generated",
                "positive": llm_summary_sentence, "positive_source": "llm_generated",
                "role": role, "judgment_id": judgment_id, "alignment_score": alignment_score,
            })
    return rows


def add_hard_negatives(df: pd.DataFrame, model_name: str, band: tuple[int, int]) -> pd.DataFrame:
    """For each row, sample a same-role, cross-source hard negative using
    similarity-rank banding (percentile band), no clustering required.

    FALLBACK: if the percentile band is empty for a given row (can happen
    with a small same-role pool, or a row whose similarities are unusually
    clustered), fall back to a uniform-random pick from the FULL same-role
    cross-source pool instead of leaving the row's negative as None. This
    matters for ablation_negatives.py, which compares an in-batch-only run
    against a hard-negative-augmented run on the SAME training rows -- if
    banding silently dropped some rows' negatives, the hard-negative
    variant would train on fewer/different rows than the in-batch variant,
    confounding "negative strategy" with "training set size" in the
    comparison. Only a genuinely tiny pool (<3 candidates) still leaves a
    row without a negative, since there's nothing sensible to fall back to.
    """
    negatives = [None] * len(df)

    for role in df["role"].unique():
        role_mask = df["role"] == role
        role_df = df[role_mask]

        for source in ["judgment_original", "llm_generated"]:
            # negative pool = the OTHER source, same role (cross-source negative)
            other_source = "llm_generated" if source == "judgment_original" else "judgment_original"
            pool = role_df[role_df["positive_source"] == other_source]["positive"].unique()
            if len(pool) < 3:
                continue
            pool_embs = embed(list(pool), model_name)

            target_idx = role_df[role_df["positive_source"] == source].index
            target_texts = role_df.loc[target_idx, "positive"].tolist()
            target_embs = embed(target_texts, model_name)

            sims = cosine_sim_matrix(target_embs, pool_embs)  # (n_targets, n_pool)
            lo_pct, hi_pct = band
            for row_i, idx in enumerate(target_idx):
                row_sims = sims[row_i]
                lo_val, hi_val = np.percentile(row_sims, [lo_pct, hi_pct])
                band_candidates = np.where((row_sims >= lo_val) & (row_sims <= hi_val))[0]
                if len(band_candidates) == 0:
                    # fallback: uniform-random from the FULL pool rather than dropping the row
                    chosen = np.random.choice(len(pool))
                else:
                    chosen = np.random.choice(band_candidates)
                negatives[df.index.get_loc(idx)] = pool[chosen]

    df["negative"] = negatives
    n_missing = df["negative"].isna().sum()
    if n_missing:
        print(f"  NOTE: {n_missing}/{len(df)} rows have no hard negative (same-role cross-source "
              f"pool had <3 candidates) -- these rows will be excluded from the hard-negative "
              f"training variant to keep both ablation variants' row counts matched.")
    return df


def main(cfg: dict, limit: int | None):
    queries_dir = Path(cfg["paths"]["stage2_dir"])
    out_dir = Path(cfg["paths"]["stage3_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = iter_judgment_ids(Path(cfg["paths"]["raw_dir"]), limit=limit)

    all_rows = []
    for jid in ids:
        path = queries_dir / f"{jid}_queries.json"
        if not path.exists():
            continue
        records = load_json(path)
        all_rows.extend(build_rows(jid, records))

    df = pd.DataFrame(all_rows)
    print(f"Built {len(df)} positive rows from {len(ids)} judgments.")

    if len(df) == 0:
        print(
            "\nERROR: 0 rows built -- there is nothing to train on.\n"
            "This means every judgment's Stage 2 query file was either missing or had zero valid\n"
            "query fields. Check the Stage 2 output above this for the actual cause (a credential\n"
            "error, rate-limiting, or every generated query failing validation are the usual reasons)\n"
            "-- Stage 3 can't produce a meaningful training table until Stage 2 actually succeeds for\n"
            f"at least one judgment. Looked for query files under: {queries_dir}"
        )
        sys.exit(1)

    if cfg["negatives"]["strategy"] == "hard_negative" and len(df) > 0:
        model_name = cfg["alignment"]["embedding_model"]
        band = tuple(cfg["negatives"]["hard_negative_percentile_band"])
        df = add_hard_negatives(df, model_name, band)
        print("Hard negatives mined (cross-source, similarity-rank banded).")
    else:
        df["negative"] = None
        print("Using in-batch negatives (Option A) — 'negative' column left empty by design.")

    out_path = out_dir / "training_table.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved training table -> {out_path}")
    print(df["positive_source"].value_counts())
    print(df["query_style"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.limit)
