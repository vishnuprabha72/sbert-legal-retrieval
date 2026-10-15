"""
find_stale_stage2_queries.py

Run AFTER regenerating Stage 1 with --force (so stage1_alignments reflects
current Stage 0 segments). Stage 2's cached output embeds its own copy of
segment_text (inherited from the alignment record it was generated from),
so when Stage 0/1 change underneath it, Stage 2's existence-only skip
check has no way to notice -- it just keeps serving stale text forever.

This compares each judgment's FRESH Stage 1 segment_text against what's
currently baked into its Stage 2 output, and flags any mismatch -- these
are the judgments whose Stage 2 output needs deleting and regenerating,
without deleting output for the (likely much larger) set of judgments
whose segments never changed and don't need a real Llama call at all.

Usage:
    uv run python find_stale_stage2_queries.py            # report only
    uv run python find_stale_stage2_queries.py --fix        # also delete stale Stage 2 output
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
    print(f"Checking {len(query_files)} Stage 2 output files against current Stage 1 alignments...")

    stale = []
    ok = 0
    missing_alignments = []

    for qpath in query_files:
        jid = qpath.stem.replace("_queries", "")
        apath = align_dir / f"{jid}_alignments.json"
        if not apath.exists():
            missing_alignments.append(jid)
            continue

        try:
            queries = json.loads(qpath.read_text())
            alignments = json.loads(apath.read_text())
        except Exception as e:
            print(f"  WARNING: couldn't read files for {jid}: {e}")
            continue

        # Build segment_id -> segment_text from the CURRENT (fresh) alignments
        current_text_by_seg = {a.get("segment_id"): a.get("segment_text") for a in alignments}

        # Compare against what's actually baked into the cached Stage 2 output
        is_stale = False
        for q in queries:
            seg_id = q.get("segment_id")
            cached_text = q.get("segment_text")
            current_text = current_text_by_seg.get(seg_id)
            if current_text is not None and current_text != cached_text:
                is_stale = True
                break
        # Also stale if the NUMBER of alignments changed (segments were re-split
        # differently, producing more/fewer aligned pairs than before)
        if len(queries) != len(alignments):
            is_stale = True

        if is_stale:
            stale.append(jid)
        else:
            ok += 1

    print(f"\n{ok} judgment(s) unchanged -- no action needed.")
    if missing_alignments:
        print(f"{len(missing_alignments)} judgment(s) have Stage 2 output but no Stage 1 alignments file "
              f"at all -- investigate separately: {missing_alignments}")

    print(f"\n{len(stale)} judgment(s) STALE (segment_text changed since Stage 2 was generated):")
    for jid in stale:
        print(f"  {jid}")

    if not stale:
        print("\nNothing to fix.")
        return

    if not fix:
        print(f"\nRun with --fix to delete these {len(stale)} judgments' Stage 2 output so they "
              f"regenerate against the current, correct segment text on your next pipeline run.")
        return

    print(f"\nDeleting Stage 2 output for {len(stale)} stale judgment(s)...")
    for jid in stale:
        f = query_dir / f"{jid}_queries.json"
        f.unlink()
        print(f"  deleted {f}")
    print(f"\nDone. These {len(stale)} judgments will make real Llama calls on your next "
          f"'run_pipeline.py' run. The other {ok} judgments are confirmed unchanged and stay cached.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fix", action="store_true", help="delete stale Stage 2 output for regeneration")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.fix)
