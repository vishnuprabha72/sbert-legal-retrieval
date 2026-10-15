"""
Stage 0 — Merge consecutive same-role paragraphs into coherent segments.

Input:  data/raw/{id}.json            (paragraphs with rule_tag/llm_tag)
Output: data/stage0_segments/{id}_segments.json

Run:
    uv run python src/stage0_merge_segments.py --limit 3
"""
import argparse
from pathlib import Path
from common.io_utils import load_config, load_judgment_paragraphs, save_json, iter_judgment_ids
from common.text_utils import chunk_text_by_words


def merge_segments(paragraphs: list[dict], role_key: str, max_words: int,
                    exclude_roles: set[str] | None = None) -> list[dict]:
    exclude_roles = exclude_roles or set()
    segments = []
    current = None
    oversized_paragraph_count = 0

    def flush():
        if current is not None:
            segments.append(current)

    for para in paragraphs:
        role = para.get(role_key)
        if role in exclude_roles:
            continue
        text = para.get("text", "").strip()
        if not text:
            continue

        para_id = para.get("paragraph_id") or para.get("label") or str(para.get("number"))

        # A single raw paragraph can already exceed max_words when upstream
        # splitting lumped an annexure or schedule into one paragraph (up to
        # ~34,000 words in our data). The merge check below only applies
        # when combining two paragraphs, so split such a paragraph into
        # sentence-packed chunks first.
        word_count = len(text.split())
        if word_count > max_words:
            oversized_paragraph_count += 1
            flush()
            current = None
            for chunk_idx, chunk_text in enumerate(chunk_text_by_words(text, max_words)):
                segments.append({
                    "role": role,
                    "text": chunk_text,
                    "paragraph_ids": [f"{para_id}_chunk{chunk_idx}"],
                })
            continue

        if current and current["role"] == role and \
           len((current["text"] + " " + text).split()) <= max_words:
            current["text"] += " " + text
            current["paragraph_ids"].append(para_id)
        else:
            flush()
            current = {
                "role": role,
                "text": text,
                "paragraph_ids": [para_id],
            }
    flush()

    # assign segment ids
    for i, seg in enumerate(segments):
        seg["segment_id"] = f"seg_{i:04d}"

    if oversized_paragraph_count:
        print(f"  NOTE: {oversized_paragraph_count} raw paragraph(s) in this judgment already exceeded "
              f"{max_words} words on their own and were split into sentence-packed chunks -- this usually "
              f"indicates an upstream paragraph-splitting issue in the source data, worth spot-checking.")

    return segments


def main(cfg: dict, limit: int | None, force: bool = False):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = Path(cfg["paths"]["stage0_dir"])
    role_key = cfg["segment_merge"]["role_key"]
    max_words = cfg["segment_merge"]["max_segment_words"]
    exclude_roles = set(cfg["segment_merge"].get("exclude_roles", []))

    ids = iter_judgment_ids(raw_dir, limit=limit)
    print(f"Found {len(ids)} judgments to process.")

    total_segments = 0
    skipped = 0
    for jid in ids:
        out_path = out_dir / f"{jid}_segments.json"
        if out_path.exists() and not force:
            skipped += 1
            continue
        paragraphs = load_judgment_paragraphs(raw_dir / f"{jid}.json")
        segments = merge_segments(paragraphs, role_key, max_words, exclude_roles)
        for seg in segments:
            seg["judgment_id"] = jid
        save_json(out_path, segments)
        total_segments += len(segments)

    print(f"Done. {len(ids)} judgments -> {total_segments} segments -> {out_dir}")
    if skipped:
        print(f"  {skipped} judgment(s) skipped (already processed -- rerun with --force to redo).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="process only N judgments (dry run)")
    parser.add_argument("--force", action="store_true", help="reprocess judgments even if output already exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.limit, force=args.force)
