"""
Learning-curve analysis: trains separate models at increasing amounts of
training data, evaluating EVERY one of them against the SAME fixed,
frozen holdout test set (see stage4_split_dataset.py's fixed-holdout
mechanism) -- this is what makes the results across different sizes
genuinely comparable, instead of each size's accuracy being partly driven
by a differently-sized/composed test set.

What this tells you (see plot's two panels):
  - Left (accuracy/MRR vs. training size): still rising steeply = you're
    underfitting / more data will keep helping. Flattening out = you're
    hitting diminishing returns from just adding more data at this model
    size/architecture -- further gains would need something other than
    volume (bigger base model, better negatives, etc.).
  - Right (style-parity gap vs. training size): whether the source-
    invariance goal (judgment_original vs llm_generated scored comparably)
    actually improves with more data, or plateaus/worsens independent of
    raw retrieval accuracy -- these are genuinely different questions and
    a data-size increase can help one without helping the other.

Sampling is NESTED/incremental: the judgment_ids used at size=25 are a
strict superset of size=10's, size=50 a superset of size=25's, etc. (a
single shuffled order of all available train judgment_ids, sizes are
prefixes of it). This makes "what changed between two adjacent points on
the curve" a clean question -- purely more data, not different data.

Input:  data/stage3_training_table/train.parquet, val.parquet, test.parquet
        (test.parquet must be the FIXED holdout -- run stage4_split_dataset.py
        first, or this reuses whatever's already there)
Output: data/models/learning_curve/size_XXXX/           (one model per size)
        data/models/learning_curve/results.csv
        data/models/learning_curve/learning_curve.png

Run:
    uv run python learning_curve.py --sizes 10 25 50 100 200 400 693
    uv run python learning_curve.py --sizes 10 25 50 --epochs 2   # quick/cheap smoke test first
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "src")
from common.io_utils import load_config  # noqa: E402
from stage5_train_sbert import train_model  # noqa: E402
from stage6_evaluate import top_k_accuracy, style_parity_diagnostic  # noqa: E402


def nested_judgment_subsets(train_df: pd.DataFrame, sizes: list[int], seed: int) -> dict[int, set]:
    """One shuffled order of all train judgment_ids, sizes are PREFIXES of
    it -- guarantees each larger size strictly contains all smaller ones."""
    all_ids = train_df["judgment_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    order = np.array(sorted(all_ids))
    rng.shuffle(order)

    subsets = {}
    for size in sizes:
        actual_size = min(size, len(order))
        if actual_size < size:
            print(f"  NOTE: requested size={size} but only {len(order)} judgments available in train.parquet "
                  f"-- using {actual_size} instead.")
        subsets[size] = set(order[:actual_size].tolist())
    return subsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sizes", type=int, nargs="+", required=True,
                         help="Training-data sizes to test, in JUDGMENT counts, e.g. --sizes 10 25 50 100 200")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override config's training.epochs for this sweep (e.g. lower for a quicker smoke test)")
    parser.add_argument("--out-dir", default="data/models/learning_curve")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    stage3_dir = Path(cfg["paths"]["stage3_dir"])
    train_df_full = pd.read_parquet(stage3_dir / "train.parquet")
    val_df = pd.read_parquet(stage3_dir / "val.parquet")
    test_df = pd.read_parquet(stage3_dir / "test.parquet").dropna(subset=["query", "positive"])

    if test_df.empty:
        raise SystemExit("test.parquet is empty -- run stage4_split_dataset.py first "
                          "(it creates the fixed holdout test set this script evaluates against).")

    print(f"Fixed holdout test set: {test_df['judgment_id'].nunique()} judgments, {len(test_df)} rows "
          f"(same test set used for EVERY size below).")

    sizes = sorted(args.sizes)
    subsets = nested_judgment_subsets(train_df_full, sizes, seed=cfg["split"]["seed"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for size in sizes:
        judgment_subset = subsets[size]
        train_df_subset = train_df_full[train_df_full["judgment_id"].isin(judgment_subset)]
        model_dir = out_dir / f"size_{size:04d}"

        print(f"\n{'=' * 80}\nTraining size={size} judgments ({len(train_df_subset)} rows) -> {model_dir}\n{'=' * 80}")

        if (model_dir / "config.json").exists():
            print(f"  Model already exists at {model_dir} -- skipping training, loading existing model. "
                  f"(delete this folder to force retraining.)")
            model = SentenceTransformer(str(model_dir))
        else:
            model = train_model(train_df_subset, val_df, cfg, output_dir=str(model_dir), verbose=True)

        print(f"Evaluating size={size} against the fixed holdout...")
        retrieval = top_k_accuracy(model, test_df, k=5)
        parity = style_parity_diagnostic(model, test_df)

        row = {
            "size_judgments": size,
            "n_train_rows": len(train_df_subset),
            "overall_accuracy": retrieval["overall_accuracy"],
            "mrr": retrieval["mrr"],
            "n_style_pairs": parity.get("n_pairs", 0),
            "style_gap_mean": parity.get("mean_gap_judgment_minus_llm"),
            "style_gap_std": parity.get("std_gap"),
        }
        results.append(row)
        print(json.dumps(row, indent=2))

        # Save the full per-size report too, not just the summary row
        with open(model_dir / "eval_report.json", "w") as f:
            json.dump({"retrieval_accuracy": retrieval, "style_parity": parity}, f, indent=2)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "results.csv", index=False)
    print(f"\nSaved results table -> {out_dir / 'results.csv'}")
    print(results_df.to_string(index=False))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results_df["size_judgments"], results_df["overall_accuracy"], marker="o", label="Top-5 accuracy")
    axes[0].plot(results_df["size_judgments"], results_df["mrr"], marker="s", label="MRR")
    axes[0].set_xlabel("Training set size (judgments)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Retrieval accuracy vs. training data size\n(evaluated on the SAME fixed holdout every time)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(results_df["size_judgments"], results_df["style_gap_mean"],
                      yerr=results_df["style_gap_std"], marker="o", capsize=4, color="#9467bd")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Training set size (judgments)")
    axes[1].set_ylabel("Mean gap (judgment_original - llm_generated)")
    axes[1].set_title("Style-parity gap vs. training data size\n(closer to 0 = better source invariance; band = ±1 std)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "learning_curve.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
