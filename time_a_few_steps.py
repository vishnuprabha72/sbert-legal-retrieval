"""
Runs a HANDFUL of real training steps (not a full epoch) using the actual
training loop's own code, just capped short, so you get real fwd/bwd/opt
timing before committing to a full run. Works for either encoding path.

Usage:
    uv run python time_a_few_steps.py --steps 5
    uv run python time_a_few_steps.py --steps 5 --legacy   # time the non-chunk-pool path instead
"""
import argparse
import sys
import time

sys.path.insert(0, "src")
import pandas as pd
from sentence_transformers import SentenceTransformer

from common.io_utils import load_config
from stage5_train_sbert import build_source_mixed_order

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--legacy", action="store_true", help="time the legacy (truncating) path instead of chunk-pool")
args = parser.parse_args()

cfg = load_config("config.yaml")
train_df = pd.read_parquet("data/stage3_training_table/train.parquet").dropna(subset=["query", "positive"])
batch_size = cfg["training"]["batch_size"]
mixed_order = build_source_mixed_order(
    sources=train_df["positive_source"].tolist(), batch_size=batch_size,
    min_fraction=cfg["training"]["min_source_fraction_per_batch"])
df = train_df.iloc[mixed_order].reset_index(drop=True)

model = SentenceTransformer(cfg["training"]["base_model"])
print(f"Model: {cfg['training']['base_model']} | max_seq_length={model.max_seq_length} | "
      f"total training rows={len(df)} | batch_size={batch_size}")

if args.legacy:
    from sentence_transformers import losses
    from torch.utils.data import DataLoader
    from stage5_train_sbert import df_to_examples
    examples = df_to_examples(df.head(batch_size * args.steps))
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: batch)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    for i, batch in enumerate(loader):
        t0 = time.time()
        features, labels = model.smart_batching_collate(batch)
        loss_val = loss_fn(features, labels)
        loss_val.backward()
        t1 = time.time()
        print(f"step {i+1}/{args.steps} total={t1-t0:.2f}s")
        if i + 1 >= args.steps:
            break
else:
    import torch
    from common.chunk_pool_encoding import ChunkPoolMultipleNegativesRankingLoss
    cp_cfg = cfg["training"].get("chunk_pool", {})
    loss_fn = ChunkPoolMultipleNegativesRankingLoss(
        model, max_tokens=cp_cfg.get("max_tokens"), pooling=cp_cfg.get("pooling", "mean"),
        scale=cp_cfg.get("scale", 20.0))
    for i in range(args.steps):
        start, end = i * batch_size, (i + 1) * batch_size
        batch_q = df["query"].tolist()[start:end]
        batch_p = df["positive"].tolist()[start:end]
        t0 = time.time()
        loss = loss_fn(batch_q, batch_p)
        t_fwd = time.time()
        loss.backward()
        t_bwd = time.time()
        print(f"step {i+1}/{args.steps} fwd={t_fwd-t0:.2f}s bwd={t_bwd-t_fwd:.2f}s total={t_bwd-t0:.2f}s")

print("\nExtrapolation: total_time_per_epoch = steps_per_epoch * avg_step_time")
print(f"steps_per_epoch for full train set = {len(df) // batch_size}")
