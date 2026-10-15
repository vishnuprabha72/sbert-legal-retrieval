"""
Visualize Stage 5 training: the loss curve (parsed from the training
console log) and per-epoch validation retrieval accuracy/MRR (parsed from
the CSV that InformationRetrievalEvaluator writes automatically).

WHY YOU NEED A LOG FILE: sentence-transformers' legacy .fit() wrapper only
ever PRINTS training loss to the console (the {'loss': '0.604', ...} lines
you see during training) -- it never saves loss history to a file on its
own. So you must capture that output yourself when running training:

    uv run python src/stage5_train_sbert.py 2>&1 | tee train_log.txt

('tee' shows the output live in your terminal AND saves it to
train_log.txt at the same time -- you don't lose the live view.)

The validation accuracy/MRR side needs no extra step -- it's already
written automatically to data/models/sbert-legal-v1/eval/*.csv by the
evaluator added in stage5_train_sbert.py.

Run:
    uv run python plot_training_curves.py --log train_log.txt
    uv run python plot_training_curves.py --log train_log.txt --model-dir data/models/sbert-legal-v1
"""
import argparse
import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe (no display needed, e.g. over SSH/WSL)
import matplotlib.pyplot as plt
import pandas as pd

# Matches the dict-repr lines HF Trainer's default logger prints, e.g.:
# {'loss': '0.6042', 'grad_norm': '6.71', 'learning_rate': '1.811e-05', 'epoch': '0.7407'}
LOSS_LINE_RE = re.compile(r"\{'loss':.*?\}")


def parse_loss_log(log_path: str) -> pd.DataFrame:
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    rows = []
    for match in LOSS_LINE_RE.finditer(text):
        try:
            d = ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError):
            continue
        try:
            rows.append({
                "epoch": float(d["epoch"]),
                "loss": float(d["loss"]),
            })
        except (KeyError, ValueError):
            continue
    if not rows:
        raise SystemExit(
            f"No loss lines found in {log_path}. Did you capture stdout when running training? "
            f"e.g.: uv run python src/stage5_train_sbert.py 2>&1 | tee train_log.txt"
        )
    return pd.DataFrame(rows)


def find_eval_csv(model_dir: str) -> Path | None:
    eval_dir = Path(model_dir) / "eval"
    if not eval_dir.exists():
        return None
    candidates = list(eval_dir.glob("Information-Retrieval_evaluation*_results.csv"))
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to the captured training stdout log")
    parser.add_argument("--model-dir", default="data/models/sbert-legal-v1",
                         help="Model output directory (where eval/ CSV lives)")
    parser.add_argument("--out", default="training_curves.png", help="Output image path")
    args = parser.parse_args()

    loss_df = parse_loss_log(args.log)
    eval_csv = find_eval_csv(args.model_dir)
    eval_df = pd.read_csv(eval_csv) if eval_csv else None

    n_plots = 2 if eval_df is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # --- Loss curve ---
    axes[0].plot(loss_df["epoch"], loss_df["loss"], marker="o", markersize=3, linewidth=1.5, color="#d62728")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3)

    # --- Validation accuracy / MRR curve ---
    if eval_df is not None:
        acc_cols = [c for c in eval_df.columns if "Accuracy@" in c or "MRR@" in c]
        for col in acc_cols:
            axes[1].plot(eval_df["epoch"], eval_df[col], marker="o", markersize=4, label=col)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Validation retrieval accuracy / MRR")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
    else:
        print(f"No validation CSV found under {args.model_dir}/eval/ -- showing loss curve only. "
              f"(Re-run stage5_train_sbert.py with the updated version that passes an evaluator "
              f"if you want this plotted too.)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot -> {args.out}")

    # Also print a quick numeric summary, useful even without opening the image
    print(f"\nLoss: {loss_df['loss'].iloc[0]:.4f} (start) -> {loss_df['loss'].iloc[-1]:.4f} (end)")
    if eval_df is not None:
        print("\nValidation scores by epoch:")
        print(eval_df[["epoch"] + acc_cols].to_string(index=False))


if __name__ == "__main__":
    main()
