"""
find_and_fix_oversized_segments.py

Scans ALL currently-present Stage 0 segment files (regardless of whether
Stage 3/4 has processed them yet, and regardless of how many judgments
that is -- 100, 693, whatever's on disk) for segments that exceed
max_segment_words by a wide margin. A segment that's, say, 310 words is
borderline-normal (a single long sentence that couldn't be split further);
a segment at 6,803 words is not borderline -- it's a sign this judgment's
Stage 0 output predates the sentence-aware oversized-paragraph splitting
fix (i.e. it's reused/pasted output from before that fix existed), not a
genuinely hard case the current code still can't handle.

For every affected judgment found, deletes its Stage 0/1/2 output files
so the next pipeline run regenerates them with the CURRENT (fixed)
splitting logic. Judgments never processed by Stage 0 at all are left
alone -- they'll be split correctly the first time the pipeline reaches
them, since the fix already lives in the current code, not something
this script needs to apply.

Usage:
    uv run python find_and_fix_oversized_segments.py             # scan AND fix
    uv run python find_and_fix_oversized_segments.py --dry-run   # scan only, print findings, delete nothing
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from common.io_utils import load_config


def main(cfg: dict, dry_run: bool):
    seg_dir = Path(cfg["paths"]["stage0_dir"])
    align_dir = Path(cfg["paths"]["stage1_dir"])
    query_dir = Path(cfg["paths"]["stage2_dir"])
    max_words = cfg["segment_merge"]["max_segment_words"]
    # Flag anything MEANINGFULLY over the cap, not borderline -- avoids false
    # positives from a segment that's slightly over due to one long sentence
    # that genuinely couldn't be split further without breaking mid-sentence.
    flag_threshold = max_words * 1.5

    seg_files = sorted(seg_dir.glob("*_segments.json"))
    print(f"Scanning {len(seg_files)} existing Stage 0 segment files on disk "
          f"(max_segment_words={max_words}, flagging anything over {flag_threshold:.0f} words)...")

    affected = []
    for path in seg_files:
        jid = path.stem.replace("_segments", "")
        try:
            segments = json.loads(path.read_text())
        except Exception as e:
            print(f"  WARNING: couldn't read {path}: {e}")
            continue
        worst = 0
        for seg in segments:
            wc = len((seg.get("text") or "").split())
            worst = max(worst, wc)
        if worst > flag_threshold:
            affected.append((jid, worst))

    affected.sort(key=lambda x: -x[1])
    print(f"\n{len(affected)} judgment(s) affected (oversized segment likely predates the Stage 0 fix):")
    for jid, wc in affected:
        print(f"  judgment {jid}: worst segment = {wc} words")

    if not affected:
        print("Nothing to fix.")
        return

    if dry_run:
        print("\n--dry-run: no files deleted. Rerun without --dry-run to actually fix these.")
        return

    print(f"\nDeleting Stage 0/1/2 output for {len(affected)} affected judgment(s) so they get "
          f"regenerated with the current (fixed) splitting logic on the next pipeline run...")
    for jid, _ in affected:
        for d, suffix in [(seg_dir, "_segments.json"), (align_dir, "_alignments.json"),
                          (query_dir, "_queries.json")]:
            f = d / f"{jid}{suffix}"
            if f.exists():
                f.unlink()
                print(f"  deleted {f}")

    print(f"\nDone. {len(affected)} judgment(s) will be freshly reprocessed on your next "
          f"'run_pipeline.py' run -- Stage 2 WILL make real Llama calls for these specific "
          f"judgments (new segment boundaries mean their old cached queries no longer match), "
          f"so make sure real credentials are in .env before rerunning. All other judgments "
          f"remain cached/untouched -- this only affects the {len(affected)} listed above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="scan and report only, don't delete anything")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.dry_run)
