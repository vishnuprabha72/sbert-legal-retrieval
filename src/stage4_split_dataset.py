"""
Stage 4 — Split the training table by judgment_id (never by row) into
train/val/test so no judgment leaks across splits.

Fixed holdout: the test set is a frozen set of judgment_ids, created once
and reused unchanged on every later run. When comparing models trained on
different amounts of data (learning_curve.py), a test set that changed
between runs would mix up "the model got better" with "the test set got
easier".

Behavior:
  - First run (no fixed_holdout_file yet): randomly samples
    split.fixed_holdout_size judgment_ids from whatever's currently in the
    training table, saves them to split.fixed_holdout_file, and uses them
    as the test set for this run.
  - Every subsequent run: loads that same file and reuses those exact
    judgment_ids as the test set, regardless of how many total judgments
    are in the training table now. Any holdout id not present in the
    current table is skipped (e.g. when only a subset of the raw data is
    being processed).
  - The remaining (non-holdout) judgments are split into train/val using
    the configured train/val ratios, renormalized to exclude the fixed
    test fraction.

Use --force-new-holdout to deliberately regenerate the holdout set (this
invalidates any past learning-curve comparisons that relied on the old
one -- only do this knowingly, e.g. at the very start of a project before
any real comparisons have been made yet).

Input:  data/stage3_training_table/training_table.parquet
Output: data/stage3_training_table/train.parquet
        data/stage3_training_table/val.parquet
        data/stage3_training_table/test.parquet
        data/fixed_holdout_judgment_ids.txt   (created once, then reused)

Run:
    uv run python src/stage4_split_dataset.py
    uv run python src/stage4_split_dataset.py --force-new-holdout   # rare, see warning above
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.io_utils import load_config


def load_or_create_holdout(holdout_path: Path, available_ids: list, holdout_size: int, seed: int,
                            force_new: bool) -> set:
    if holdout_path.exists() and not force_new:
        ids = set(line.strip() for line in holdout_path.read_text().splitlines() if line.strip())
        present = ids & set(available_ids)
        missing = ids - present
        print(f"Loaded FIXED holdout from {holdout_path}: {len(ids)} judgment_ids "
              f"({len(present)} present in current data, {len(missing)} not found in current data -- skipped).")
        return present

    if holdout_path.exists() and force_new:
        print(f"--force-new-holdout: overwriting existing holdout at {holdout_path} "
              f"(any past learning-curve comparisons using the old holdout are no longer comparable to future runs).")

    rng = np.random.default_rng(seed)
    available = np.array(sorted(available_ids))
    size = min(holdout_size, len(available))
    chosen = set(rng.choice(available, size=size, replace=False).tolist())

    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text("\n".join(sorted(chosen)) + "\n")
    print(f"Created NEW fixed holdout ({len(chosen)} judgment_ids) -> {holdout_path}. "
          f"This file will be reused unchanged on every future run unless you pass --force-new-holdout.")
    return chosen


def main(cfg: dict, force_new_holdout: bool = False):
    stage3_dir = Path(cfg["paths"]["stage3_dir"])
    df = pd.read_parquet(stage3_dir / "training_table.parquet")

    all_judgment_ids = df["judgment_id"].unique().tolist()

    holdout_path = Path(cfg["split"]["fixed_holdout_file"])
    test_ids = load_or_create_holdout(
        holdout_path=holdout_path,
        available_ids=all_judgment_ids,
        holdout_size=cfg["split"]["fixed_holdout_size"],
        seed=cfg["split"]["seed"],
        force_new=force_new_holdout,
    )

    remaining_ids = [jid for jid in all_judgment_ids if jid not in test_ids]

    # Renormalize train/val ratios over the remaining (non-holdout) pool --
    # e.g. configured train=0.8/val=0.1/test=0.1 becomes an 8:1 split of
    # whatever's left after the fixed test set is removed.
    train_frac = cfg["split"]["train"]
    val_frac = cfg["split"]["val"]
    val_frac_of_remainder = val_frac / (train_frac + val_frac) if (train_frac + val_frac) > 0 else 0.0

    rng = np.random.default_rng(cfg["split"]["seed"])
    remaining_arr = np.array(remaining_ids)
    rng.shuffle(remaining_arr)

    n_val = int(len(remaining_arr) * val_frac_of_remainder)
    val_ids = set(remaining_arr[:n_val].tolist())
    train_ids = set(remaining_arr[n_val:].tolist())

    train_df = df[df["judgment_id"].isin(train_ids)]
    val_df = df[df["judgment_id"].isin(val_ids)]
    test_df = df[df["judgment_id"].isin(test_ids)]

    train_df.to_parquet(stage3_dir / "train.parquet", index=False)
    val_df.to_parquet(stage3_dir / "val.parquet", index=False)
    test_df.to_parquet(stage3_dir / "test.parquet", index=False)

    print(f"Judgments -> train:{len(train_ids)} val:{len(val_ids)} test:{len(test_ids)} (fixed)")
    print(f"Rows      -> train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force-new-holdout", action="store_true",
                         help="Regenerate the fixed holdout set (invalidates past learning-curve "
                              "comparisons -- see module docstring)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, force_new_holdout=args.force_new_holdout)
