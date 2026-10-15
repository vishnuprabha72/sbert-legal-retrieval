"""
find_empty_stage2_outputs.py

Stage 2's resumability check is existence-based ("does {id}_queries.json
exist"), not content-based -- an empty [] file looks exactly like "already
done" and gets skipped forever, silently contributing zero training rows
for that judgment with no error or warning.

An empty Stage 2 output has two genuinely different causes that need
different responses:
  1. LEGITIMATE: Stage 1 found zero acceptable alignments for that judgment
     (nothing passed the cosine/BM25 gate) -- Stage 2 correctly has
     nothing to generate from. [] is the correct, final output here.
  2. NEEDS REGENERATION: Stage 1 DID find alignments, but Stage 2 still
     ended up empty -- something failed (a stale/incomplete pasted file,
     or a genuine generation failure that wasn't logged as such). This
     needs to be deleted and regenerated.

This script finds every empty stage2_queries file and cross-checks it
against the corresponding stage1_alignments file to sort them into the
two buckets, so you only spend real Llama calls on bucket 2.

Usage:
    uv run python find_empty_stage2_outputs.py             # report only
    uv run python find_empty_stage2_outputs.py --fix        # also delete bucket-2 files for regeneration
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from common.io_utils import load_config


def main(cfg: dict, fix: bool):
    align_dir = Path(cfg["paths"]["stage1_dir"])
    query_dir = Path(cfg["paths"]["stage2_dir"])

    query_files = sorted(query_dir.glob("*_queries.json"))
    print(f"Checking {len(query_files)} existing Stage 2 output files for empty results...")

    legitimate = []
    needs_regen = []
    unreadable = []

    for qpath in query_files:
        jid = qpath.stem.replace("_queries", "")
        try:
            queries = json.loads(qpath.read_text())
        except Exception as e:
            unreadable.append((jid, str(e)))
            continue
        if queries:   # non-empty, nothing to check
            continue

        apath = align_dir / f"{jid}_alignments.json"
        if not apath.exists():
            needs_regen.append(jid)   # no alignment file at all -- definitely wrong, investigate
            continue
        try:
            alignments = json.loads(apath.read_text())
        except Exception as e:
            unreadable.append((jid, f"alignments file unreadable: {e}"))
            continue

        if len(alignments) == 0:
            legitimate.append(jid)
        else:
            needs_regen.append(jid)

    print(f"\n{len(legitimate)} legitimate zero-alignment judgment(s) (Stage 1 also found nothing -- "
          f"correct, no action needed):")
    for jid in legitimate:
        print(f"  {jid}")

    print(f"\n{len(needs_regen)} judgment(s) NEED REGENERATION (Stage 1 found real alignments, "
          f"but Stage 2 output is empty anyway):")
    for jid in needs_regen:
        print(f"  {jid}")

    if unreadable:
        print(f"\n{len(unreadable)} file(s) couldn't be read/parsed at all -- investigate manually:")
        for jid, err in unreadable:
            print(f"  {jid}: {err}")

    if not needs_regen:
        print("\nNothing to fix.")
        return

    if not fix:
        print(f"\nRun with --fix to delete these {len(needs_regen)} judgments' Stage 2 output "
              f"so they get properly regenerated (real Llama calls) on your next pipeline run.")
        return

    print(f"\nDeleting Stage 2 output for {len(needs_regen)} judgment(s)...")
    for jid in needs_regen:
        f = query_dir / f"{jid}_queries.json"
        f.unlink()
        print(f"  deleted {f}")
    print(f"\nDone. These {len(needs_regen)} judgments will make real Llama calls on your next "
          f"'run_pipeline.py' run. Everything else -- including the {len(legitimate)} legitimate "
          f"zero-alignment cases -- is untouched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fix", action="store_true", help="delete Stage 2 output for judgments that need regeneration")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.fix)
