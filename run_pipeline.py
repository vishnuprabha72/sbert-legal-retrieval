"""
Orchestrator — runs pipeline stages 0-4 in order (data prep only; training/eval
are run separately since they're longer and you'll want to inspect the dataset
first).

Dry run on 3 judgments:
    uv run python run_pipeline.py --limit 3

Full run:
    uv run python run_pipeline.py

Then, once you're happy with the dataset:
    uv run python src/stage5_train_sbert.py
    uv run python src/stage6_evaluate.py
"""
import argparse
import subprocess
import sys

sys.path.insert(0, "src")
from common.check_env import check_credentials  # noqa: E402

DATA_PREP_STAGES = [
    "src/stage0_merge_segments.py",
    "src/stage1_align_summaries.py",
    "src/stage2_llama_generate.py",
    "src/stage3_build_training_table.py",
]

FINAL_STAGE_NO_LIMIT = "src/stage4_split_dataset.py"


def run(limit: int | None):
    # Fail fast: check credentials BEFORE running stage 0/1, which cost real
    # time (embeddings, BM25) but no API money -- no point burning minutes on
    # those only to hit an auth error at stage 2 that a 1-second check here
    # would have caught immediately.
    check_credentials()

    for stage in DATA_PREP_STAGES:
        cmd = ["uv", "run", "python", stage]
        if limit:
            cmd += ["--limit", str(limit)]
        print(f"\n=== Running {stage} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Stage failed: {stage}")
            sys.exit(1)

    print(f"\n=== Running {FINAL_STAGE_NO_LIMIT} ===")
    result = subprocess.run(["uv", "run", "python", FINAL_STAGE_NO_LIMIT])
    if result.returncode != 0:
        print(f"Stage failed: {FINAL_STAGE_NO_LIMIT}")
        sys.exit(1)

    print("\nData preparation pipeline complete.")
    print("Next: uv run python src/stage5_train_sbert.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process only N judgments (dry run)")
    args = parser.parse_args()
    run(args.limit)
