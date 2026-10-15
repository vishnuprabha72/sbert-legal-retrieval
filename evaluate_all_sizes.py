"""
Runs the FULL stage6_evaluate.py report (by-role, by-query-style, style-
parity) against every data/models/learning_curve/size_XXXX/ model, and
combines them into one comparison table -- learning_curve.py's own
results.csv only has overall accuracy/MRR/style-gap, not the per-role
breakdown, so this fills that gap (e.g. "does the LAW role peak at a
different training size than FACT?").

Run:
    uv run python evaluate_all_sizes.py
    uv run python evaluate_all_sizes.py --learning-curve-dir data/models/learning_curve
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from common.io_utils import load_config  # noqa: E402
from stage6_evaluate import main as evaluate_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--learning-curve-dir", default="data/models/learning_curve")
    parser.add_argument("--out", default="data/models/learning_curve/by_role_comparison.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    lc_dir = Path(args.learning_curve_dir)
    size_dirs = sorted(lc_dir.glob("size_*"), key=lambda p: int(p.name.split("_")[1]))

    if not size_dirs:
        raise SystemExit(f"No size_XXXX folders found under {lc_dir} -- run learning_curve.py first.")

    rows = []
    for size_dir in size_dirs:
        if not (size_dir / "config.json").exists():
            print(f"Skipping {size_dir} -- doesn't look like a complete model folder.")
            continue

        size = int(size_dir.name.split("_")[1])
        print(f"\n{'=' * 60}\nEvaluating {size_dir.name}\n{'=' * 60}")
        report = evaluate_model(cfg, model_path=str(size_dir))

        row = {"size_judgments": size,
               "overall_accuracy": report["retrieval_accuracy"]["overall_accuracy"],
               "mrr": report["retrieval_accuracy"]["mrr"]}
        for role, acc in report["retrieval_accuracy"]["by_role"].items():
            row[f"role_{role}"] = acc
        for style, acc in report["retrieval_accuracy"]["by_query_style"].items():
            row[f"style_{style}"] = acc
        row["style_gap_mean"] = report["style_parity"]["mean_gap_judgment_minus_llm"]
        row["style_gap_std"] = report["style_parity"]["std_gap"]
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values("size_judgments")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.out, index=False)

    print(f"\n\nSaved combined by-role comparison -> {args.out}")
    print(result_df.to_string(index=False))

    # Quick flag: for each role, which size scored best?
    role_cols = [c for c in result_df.columns if c.startswith("role_")]
    print("\nBest size per role:")
    for col in role_cols:
        best_row = result_df.loc[result_df[col].idxmax()]
        print(f"  {col.replace('role_', ''):10s}: size={int(best_row['size_judgments'])} (score={best_row[col]:.4f})")


if __name__ == "__main__":
    main()
