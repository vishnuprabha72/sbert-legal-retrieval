"""
Stage 5 — Fine-tune SBERT (all-MiniLM by default) using MultipleNegativesRankingLoss.

Uses in-batch negatives (Option A). Training examples are PRE-ORDERED so that
every consecutive batch-sized chunk mixes both positive_source values
(judgment_original / llm_generated) instead of relying on plain shuffling to
achieve this by chance -- this is what preserves the cross-source contrastive
signal needed for style-invariance.

The mixing is done once, by reordering the examples, and the data is then
fed through a plain DataLoader with shuffle=False. A custom batch_sampler
would be simpler but does not work with the legacy `.fit()` wrapper in
sentence-transformers 5.x, which reads `data_loader.batch_size` (None when a
batch_sampler is given).

VALIDATION DURING TRAINING: a retrieval-style evaluator runs against
val.parquet at the end of every epoch, using InformationRetrievalEvaluator
(queries -> corpus -> relevant doc ids), NOT EmbeddingSimilarityEvaluator
(that's for STS-style similarity-score pairs, which isn't what this
dataset is). Both judgment_original and llm_generated positives for the
same query are marked relevant, matching the dual-positive/source-invariance
design goal. With an evaluator passed to model.fit(), `save_best_model=True`
(the default) means the FINAL saved model is whichever epoch scored best
on val-set retrieval accuracy, not necessarily the last epoch trained.

CHECKPOINTING (optional, off by default -- see config.yaml's
training.checkpointing): separate from the "best model" above, this saves
periodic intermediate snapshots (every N steps) so you can inspect or
resume from a specific point during training, not just the single
best-scoring epoch.

CHUNK-POOL TRAINING (config.yaml's training.chunk_pool.enabled, default
true): model.fit() truncates every training text at max_seq_length (256
tokens), so the tail of a long paragraph contributes no training signal. When enabled, this file bypasses model.fit()
entirely and runs a manual training loop using
common.chunk_pool_encoding.ChunkPoolMultipleNegativesRankingLoss, which
splits any oversized text into token-bounded chunks, embeds each chunk
WITH gradients, and mean-pools them into that text's training embedding
-- so the full paragraph contributes, not just its first ~150-200 words.
Validation during training is scored the same way (evaluate_retrieval_
chunk_pool), so best-epoch selection isn't silently judging the model on
truncated text either. Set chunk_pool.enabled: false to fall back to the
truncating model.fit() path, which the paper reports as Legacy-trained
(see chunk_pool_encoding.py for why model.fit() cannot be patched
instead). Chunk-pool training is slower: a 3-chunk paragraph costs three
forward passes instead of one.

The training logic lives in train_model(), which takes train_df/val_df/output_dir as plain arguments instead of
reading them from disk itself. This is what lets learning_curve.py reuse
the exact same training code to train multiple models at different data
sizes without duplicating this file's logic.

Input:  data/stage3_training_table/train.parquet, val.parquet
Output: data/models/sbert-legal-v1/  (fine-tuned model checkpoint -- the
        BEST epoch by val accuracy, not necessarily the last one trained)
        data/models/sbert-legal-v1/eval/  (per-epoch validation CSV logs)
        data/models/sbert-legal-v1/checkpoints/  (if checkpointing.enabled)

Run:
    uv run python src/stage5_train_sbert.py
    uv run python src/stage5_train_sbert.py 2>&1 | tee train_log.txt   # to plot loss curves later
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from common.chunk_pool_encoding import ChunkPoolMultipleNegativesRankingLoss, evaluate_retrieval_chunk_pool
from common.io_utils import load_config


def build_source_mixed_order(sources: list[str], batch_size: int, min_fraction: float = 0.3) -> list[int]:
    """
    Returns a REORDERED list of indices (0..len(sources)-1) such that, when
    read off in fixed-size consecutive chunks of `batch_size` (no shuffling
    needed downstream), each chunk contains at least `min_fraction` of every
    positive_source value present in `sources`.
    """
    by_source: dict[str, list[int]] = {}
    for i, s in enumerate(sources):
        by_source.setdefault(s, []).append(i)

    pools = {k: v.copy() for k, v in by_source.items()}
    for pool in pools.values():
        random.shuffle(pool)

    source_names = list(pools.keys())
    n_batches = math.ceil(len(sources) / batch_size)
    min_per_source = max(1, int(batch_size * min_fraction / len(source_names))) if source_names else 0

    order: list[int] = []
    for _ in range(n_batches):
        batch = []
        for s in source_names:
            take = min(min_per_source, len(pools[s]))
            batch.extend(pools[s][:take])
            pools[s] = pools[s][take:]
        remaining_pool = [i for p in pools.values() for i in p]
        random.shuffle(remaining_pool)
        fill_needed = batch_size - len(batch)
        batch.extend(remaining_pool[:fill_needed])
        used = set(remaining_pool[:fill_needed])
        for s in pools:
            pools[s] = [i for i in pools[s] if i not in used]
        if batch:
            random.shuffle(batch)
            order.extend(batch)
    return order


def df_to_examples(df: pd.DataFrame) -> list[InputExample]:
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(texts=[row["query"], row["positive"]]))
    return examples


def _build_eval_sets(val_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Builds the (queries, corpus, relevant_docs) dicts shared by both the
    legacy InformationRetrievalEvaluator path and the chunk-pool
    evaluator (evaluate_retrieval_chunk_pool) -- same dual-positive
    design either way: both judgment_original AND llm_generated
    positives for the same query are marked relevant.
    """
    val_df = val_df.dropna(subset=["query", "positive"])

    unique_positives = val_df["positive"].unique().tolist()
    doc_id_of = {text: f"doc_{i}" for i, text in enumerate(unique_positives)}
    corpus = {doc_id: text for text, doc_id in doc_id_of.items()}

    unique_queries = val_df["query"].unique().tolist()
    qid_of = {text: f"q_{i}" for i, text in enumerate(unique_queries)}
    queries = {qid: text for text, qid in qid_of.items()}

    relevant_docs: dict[str, set[str]] = {qid: set() for qid in queries}
    for _, row in val_df.iterrows():
        qid = qid_of[row["query"]]
        did = doc_id_of[row["positive"]]
        relevant_docs[qid].add(did)

    return queries, corpus, relevant_docs


def build_ir_evaluator(val_df: pd.DataFrame, name: str = "val") -> InformationRetrievalEvaluator:
    """
    Builds a retrieval-style validation evaluator from val.parquet, for
    the LEGACY (chunk_pool.enabled=false) training path. NOTE: this
    evaluator calls model.encode() internally and will truncate any long
    corpus paragraph at max_seq_length just like training does in that
    path -- consistent with that path's behavior, but see
    evaluate_retrieval_chunk_pool for the chunk-pool-consistent version
    used when chunk_pool.enabled=true.
    """
    queries, corpus, relevant_docs = _build_eval_sets(val_df)
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True,
        accuracy_at_k=[1, 3, 5, 10],
        mrr_at_k=[10],
    )


def _append_eval_log(output_dir: str, epoch: int, steps_per_epoch: int, metrics: dict, name: str = "val") -> None:
    """
    Appends one epoch's chunk-pool validation metrics to a CSV in the
    SAME location/naming convention sentence-transformers'
    InformationRetrievalEvaluator uses (data/models/.../eval/
    Information-Retrieval_evaluation_{name}_results.csv, with
    "...-Accuracy@k" / "...-MRR@10" columns) so existing tooling
    (plot_training_curves.py, evaluate_all_sizes.py) keeps working
    unchanged against chunk-pool-trained models.
    """
    eval_dir = Path(output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / f"Information-Retrieval_evaluation_{name}_results.csv"

    row = {"epoch": epoch, "steps": steps_per_epoch}
    for k in (1, 3, 5, 10):
        row[f"cosine-Accuracy@{k}"] = metrics.get(f"accuracy@{k}")
    row["cosine-MRR@10"] = metrics["mrr@10"]
    row["cosine-NDCG@10"] = metrics["ndcg@10"]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict, output_dir: str,
                 verbose: bool = True, use_hard_negatives: bool = None) -> SentenceTransformer:
    """
    The reusable core of Stage 5: fine-tunes a model on the given train_df,
    validates against val_df each epoch, saves the best-on-val model to
    output_dir, and returns the trained model object.

    Used both by this file's own main() (full training run) and by
    learning_curve.py (repeated calls at different train_df sizes, same
    val_df/config each time).

    use_hard_negatives: only meaningful when chunk_pool.enabled=true.
    None (default) reads config's negatives.strategy (True if
    "hard_negative", else False) -- pass an explicit True/False to
    override config, which is how ablation_negatives.py trains both
    variants from the same function without editing config.yaml between
    calls.
    """
    train_df = train_df.dropna(subset=["query", "positive"])
    if verbose:
        print(f"Training rows: {len(train_df)} | positive_source distribution:")
        print(train_df["positive_source"].value_counts())

    model = SentenceTransformer(cfg["training"]["base_model"])

    batch_size = cfg["training"]["batch_size"]
    mixed_order = build_source_mixed_order(
        sources=train_df["positive_source"].tolist(),
        batch_size=batch_size,
        min_fraction=cfg["training"]["min_source_fraction_per_batch"],
    )
    ordered_df = train_df.iloc[mixed_order].reset_index(drop=True)

    if cfg["training"].get("chunk_pool", {}).get("enabled", False):
        if use_hard_negatives is None:
            use_hard_negatives = cfg["negatives"]["strategy"] == "hard_negative"
        if use_hard_negatives:
            missing = ordered_df["negative"].isna().sum() if "negative" in ordered_df.columns else len(ordered_df)
            if missing:
                if verbose:
                    print(f"  Dropping {missing} rows with no mined hard negative before training "
                          f"(use_hard_negatives=True requires a negative for every row).")
                ordered_df = ordered_df.dropna(subset=["negative"]).reset_index(drop=True)
        return _train_model_chunk_pool(ordered_df, val_df, cfg, model, batch_size, output_dir, verbose,
                                        use_hard_negatives=use_hard_negatives)

    return _train_model_legacy(ordered_df, val_df, cfg, model, batch_size, output_dir, verbose)


def _train_model_legacy(ordered_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict, model: SentenceTransformer,
                         batch_size: int, output_dir: str, verbose: bool = True) -> SentenceTransformer:
    """
    Original model.fit()-based training path (plain truncation at
    max_seq_length -- see chunk_pool_encoding.py's module docstring for
    why this truncates and what that costs). Extracted into its own
    function, mirroring _train_model_chunk_pool, specifically so
    ablation_encoding.py can call this and the chunk-pool path directly
    with the SAME ordered_df/seed -- isolating "which encoding method" as
    the only variable between them, the same design already used for the
    in-batch-vs-hard-negative ablation.
    """
    train_examples = df_to_examples(ordered_df)

    class ListDataset:
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    train_dataloader = DataLoader(
        ListDataset(train_examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=model.smart_batching_collate,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(train_dataloader) * cfg["training"]["epochs"] * cfg["training"]["warmup_ratio"])

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    val_df_clean = val_df.dropna(subset=["query", "positive"]) if val_df is not None else pd.DataFrame()
    evaluator = build_ir_evaluator(val_df_clean) if len(val_df_clean) > 0 else None
    if evaluator is None and verbose:
        print("  WARNING: val set is empty -- skipping validation, save_best_model will have no effect.")

    ckpt_cfg = cfg["training"].get("checkpointing", {})
    checkpoint_path = ckpt_cfg.get("checkpoint_dir") if ckpt_cfg.get("enabled") else None
    if checkpoint_path:
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    fit_kwargs = dict(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=cfg["training"]["epochs"],
        evaluation_steps=0,   # 0 = evaluate at the end of every epoch only
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(cfg["training"]["learning_rate"])},
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=verbose,
    )
    if checkpoint_path:
        fit_kwargs.update(
            checkpoint_path=checkpoint_path,
            checkpoint_save_steps=ckpt_cfg.get("save_steps", 500),
            checkpoint_save_total_limit=ckpt_cfg.get("save_total_limit", 3),
        )

    model.fit(**fit_kwargs)

    if verbose:
        print(f"Training complete. Best-on-val model saved to {output_dir}")
        if evaluator is not None:
            print(f"Per-epoch validation scores logged to: "
                  f"{output_dir}/eval/Information-Retrieval_evaluation_{evaluator.name}_results.csv")
        if checkpoint_path:
            print(f"Periodic checkpoints saved to: {checkpoint_path}")

    return model


def _train_model_chunk_pool(ordered_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict, model: SentenceTransformer,
                             batch_size: int, output_dir: str, verbose: bool = True,
                             use_hard_negatives: bool = False, pooling: str = None) -> SentenceTransformer:
    """
    Manual training loop used when training.chunk_pool.enabled=true.
    Bypasses model.fit() (which has no hook for per-chunk gradient
    pooling -- see chunk_pool_encoding.py's module docstring) and instead
    drives ChunkPoolMultipleNegativesRankingLoss directly: standard
    AdamW + linear warmup schedule, matching config.yaml's existing
    learning_rate/warmup_ratio/epochs semantics from the legacy path.

    use_hard_negatives: if True, ordered_df must have a non-null
    'negative' column (see stage3_build_training_table.py's
    add_hard_negatives) and each batch's mined hard negatives are passed
    to the loss alongside in-batch negatives. If False, 'negative' is
    ignored even if present -- pure in-batch negatives, matching the
    original design. Exposed as a parameter (not just read from config)
    so ablation_negatives.py can call this twice with the same
    ordered_df/config and only this one flag differing.
    """
    cp_cfg = cfg["training"].get("chunk_pool", {})
    max_tokens = cp_cfg.get("max_tokens") or model.max_seq_length
    pooling = pooling if pooling is not None else cp_cfg.get("pooling", "mean")
    scale = cp_cfg.get("scale", 20.0)

    if use_hard_negatives:
        if "negative" not in ordered_df.columns or ordered_df["negative"].isna().any():
            raise ValueError(
                "use_hard_negatives=True but ordered_df's 'negative' column is missing or has nulls. "
                "Run stage3_build_training_table.py with negatives.strategy: hard_negative first, or "
                "drop null-negative rows before calling train_model (ablation_negatives.py does this)."
            )
        negatives_all = ordered_df["negative"].tolist()
    else:
        negatives_all = None

    queries_all = ordered_df["query"].tolist()
    positives_all = ordered_df["positive"].tolist()
    n = len(queries_all)

    epochs = cfg["training"]["epochs"]
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    if verbose:
        print(f"[chunk_pool] max_tokens={max_tokens} pooling={pooling!r} scale={scale} "
              f"hard_negatives={use_hard_negatives} steps/epoch={steps_per_epoch} "
              f"total_steps={total_steps} warmup_steps={warmup_steps}")

    loss_fn = ChunkPoolMultipleNegativesRankingLoss(model, max_tokens=max_tokens, pooling=pooling, scale=scale)
    optimizer = AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)

    val_df_clean = val_df.dropna(subset=["query", "positive"]) if val_df is not None else pd.DataFrame()
    eval_sets = _build_eval_sets(val_df_clean) if len(val_df_clean) > 0 else None
    if eval_sets is None and verbose:
        print("  WARNING: val set is empty -- skipping validation; the LAST epoch's model will be saved "
              "(save_best_model has no effect without a val set, same caveat as the legacy path).")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_score = -1.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            start, end = step * batch_size, min((step + 1) * batch_size, n)
            batch_q = queries_all[start:end]
            batch_p = positives_all[start:end]
            batch_n = negatives_all[start:end] if negatives_all is not None else None

            if verbose and epoch == 0 and step == 0:
                # Heartbeat: nothing else prints until a full forward+backward+optimizer
                # step completes, and the very first step (before caches warm up, and
                # whatever paragraph lengths this batch happens to draw) is often the
                # slowest -- without this line, a genuinely-working-but-slow first step
                # looks identical to a hang. Printed once, not every step.
                print(f"  Starting step 1/{steps_per_epoch} (first step -- may take a while "
                      f"on CPU depending on how many chunks this batch's paragraphs need)...")

            t_step_start = time.time()
            optimizer.zero_grad()
            loss = loss_fn(batch_q, batch_p, negatives=batch_n)
            t_forward = time.time()
            loss.backward()
            t_backward = time.time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            t_step_end = time.time()

            loss_val = loss.item()
            epoch_loss += loss_val
            if verbose:
                # Two lines, deliberately: the dict-repr line matches the exact pattern
                # plot_training_curves.py's regex parser looks for ({'loss': ..., 'epoch':
                # ...}), now printed every step instead of every 20 so a silent multi-step
                # gap doesn't look identical to a hang. The second line adds per-phase
                # timing so a genuinely slow step is diagnosable (which phase costs what)
                # instead of just "training is slow" with no further information.
                print({"loss": f"{loss_val:.4f}", "epoch": f"{epoch + (step + 1) / steps_per_epoch:.4f}"})
                print(f"  step {step + 1}/{steps_per_epoch} "
                      f"[fwd={t_forward - t_step_start:.1f}s bwd={t_backward - t_forward:.1f}s "
                      f"opt={t_step_end - t_backward:.1f}s total={t_step_end - t_step_start:.1f}s]")

        avg_loss = epoch_loss / steps_per_epoch
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} avg training loss: {avg_loss:.4f}")

        if eval_sets is not None:
            queries, corpus, relevant_docs = eval_sets
            metrics = evaluate_retrieval_chunk_pool(model, queries, corpus, relevant_docs,
                                                      max_tokens=max_tokens, pooling=pooling)
            if verbose:
                print(f"  val: ndcg@10={metrics['ndcg@10']:.4f} mrr@10={metrics['mrr@10']:.4f} "
                      f"acc@1={metrics['accuracy@1']:.4f} acc@10={metrics['accuracy@10']:.4f}")
            _append_eval_log(output_dir, epoch + 1, steps_per_epoch, metrics)
            if metrics["primary_score"] > best_score:
                best_score = metrics["primary_score"]
                model.save(output_dir)
                if verbose:
                    print(f"  -> new best (ndcg@10={best_score:.4f}), saved to {output_dir}")
        else:
            model.save(output_dir)

    if verbose:
        print(f"Training complete. Best-on-val model saved to {output_dir}")
        if eval_sets is not None:
            print(f"Per-epoch validation scores logged to: "
                  f"{output_dir}/eval/Information-Retrieval_evaluation_val_results.csv")

    # Reload from disk so the returned in-memory model matches the best-saved checkpoint
    # (mirrors save_best_model=True's effect in the legacy .fit()-based path).
    return SentenceTransformer(output_dir)


def main(cfg: dict):
    stage3_dir = Path(cfg["paths"]["stage3_dir"])
    train_df = pd.read_parquet(stage3_dir / "train.parquet")
    val_df = pd.read_parquet(stage3_dir / "val.parquet")

    train_model(train_df, val_df, cfg, output_dir=cfg["training"]["output_dir"], verbose=True)

    print(f"Val set has {len(val_df)} rows — run stage6_evaluate.py next for the full held-out test-set report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
