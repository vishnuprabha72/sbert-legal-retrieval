"""
Stage 1 — Align gold-summary sentences to their best-matching segment.

Every summary sentence is searched against every segment of its judgment,
not only segments with the same rhetorical role. Judgments do not keep roles
apart cleanly: courts cite statute numbers inside FACT narrative and restate
legal principles inside REASON (e.g. "Section 2(6)" may appear in a
REASON-tagged paragraph rather than the LAW paragraph), so a same-role
restriction rejects correct matches. The segment's own rhetorical role is still
recorded (`segment_role`) and given a small ranking bonus when it matches
the summary sentence's expected role (`summary_role`) -- a nudge, not a gate.

Scoring is hybrid:
  - cosine similarity (dense embeddings)      -- catches paraphrase/semantic match
  - BM25 (lexical term-overlap)                -- catches exact legal tokens
    (section numbers, citations, party/statute names) embeddings underweight
  - combined via Reciprocal Rank Fusion (RRF), so no fragile score-weight
    tuning is needed; see common/bm25_utils.py for details.

Acceptance gate: a candidate is kept if EITHER its cosine score clears
`alignment.threshold` OR its raw BM25 score clears `alignment.bm25_min_score`
-- this lets strong lexical matches (statute citations) through even when
they're semantically "distant" prose-wise, and vice versa.

Reads the summary_field_specs from config.yaml to handle the real gold
summary schema, which mixes:
  - flat strings                (e.g. "Facts")
  - nested objects               (e.g. "Arguments" -> "Petitioner"/"Respondent")
  - lists of strings             (e.g. "Issues or Questions")
  - lists of objects             (e.g. "Statutes Referred" -> concat Name+Principle per item)

Input:  data/stage0_segments/{id}_segments.json
        data/raw/{id}_corrected_summary.json
Output: data/stage1_alignments/{id}_alignments.json
        data/stage1_alignments/_audit_sample.json   (random sample for manual spot-check)

Run:
    uv run python src/stage1_align_summaries.py --limit 3
"""
import argparse
import random
import re
from pathlib import Path

import numpy as np

from common.io_utils import load_config, load_json, load_gold_summary, save_json, iter_judgment_ids
from common.embedding_utils import embed, cosine_sim_matrix
from common.bm25_utils import build_bm25, bm25_scores_for_query, rrf_fuse

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Common abbreviations that shouldn't trigger a sentence break, especially
# frequent in Indian legal citations (e.g. "S.B. Sardar Ranjit Singh",
# "Associated Hotels ... v. Abdul Alim", "M.R. Shah, J.")
ABBREVIATIONS = ["Mr", "Mrs", "Dr", "No", "Sec", "Art", "Ord", "Ltd", "Co",
                  "etc", "vs", "v", "Jj", "J", "Ors", "Anr", "Smt"]


def _coerce_to_text(value) -> str:
    """
    Some gold-summary JSON files have schema drift across the corpus --
    a field like "Facts" or "Arguments" -> "Petitioner" is USUALLY a plain
    string, but for a handful of judgments (observed: ~11 out of 693) it's
    actually a list or dict instead. Rather than crash that judgment
    entirely (AttributeError: 'list'/'dict' object has no attribute
    'strip'), best-effort-flatten it into text so alignment can still
    proceed for whatever content IS extractable.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_coerce_to_text(v).rstrip(".") for v in value]
        return ". ".join(p for p in parts if p)
    if isinstance(value, dict):
        # join all string-like values found in the dict, shallow
        parts = [_coerce_to_text(v).rstrip(".") for v in value.values()]
        return ". ".join(p for p in parts if p)
    if value is None:
        return ""
    return str(value)


def split_sentences(text) -> list[str]:
    text = _coerce_to_text(text).strip()
    if not text:
        return []
    protected = text
    # protect single-capital-letter initials, e.g. "S.B.", "G.K.", "M.R."
    protected = re.sub(r"\b([A-Z])\.(?=\s*[A-Z]\.|\s+[A-Z][a-z])", r"\1<DOT>", protected)
    # protect known legal/common abbreviations
    for abbr in ABBREVIATIONS:
        protected = re.sub(rf"\b{re.escape(abbr)}\.", f"{abbr}<DOT>", protected)
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(protected) if s.strip()]
    return [s.replace("<DOT>", ".") for s in sentences]


def extract_sentences_for_role(summary: dict, spec: dict) -> list[str]:
    """Pull sentences out of the gold summary for one field spec (string /
    nested_string / list / list_of_objects)."""
    field_type = spec["type"]

    if field_type == "string":
        return split_sentences(summary.get(spec["key"], ""))

    if field_type == "nested_string":
        nested = summary.get(spec["key"], {})
        if not isinstance(nested, dict):
            return []
        return split_sentences(nested.get(spec["subkey"], ""))

    if field_type == "list":
        items = summary.get(spec["key"], [])
        if not isinstance(items, list):
            return []
        sentences = []
        for item in items:
            if isinstance(item, str):
                sentences.extend(split_sentences(item))
        return sentences

    if field_type == "list_of_objects":
        items = summary.get(spec["key"], [])
        if not isinstance(items, list):
            return []
        sentences = []
        for item in items:
            if not isinstance(item, dict):
                continue
            parts = [str(item.get(f, "")) for f in spec.get("fields", []) if item.get(f)]
            combined = ". ".join(p.rstrip(".") for p in parts if p)
            if combined:
                sentences.append(combined)
        return sentences

    return []


def spec_label(spec: dict) -> str:
    """Human-readable source-field label for audit/debug, e.g.
    'Statutes Referred' or 'Arguments.Petitioner'."""
    if spec["type"] == "nested_string":
        return f'{spec["key"]}.{spec["subkey"]}'
    return spec["key"]


def collect_candidate_sentences(summary: dict, field_specs: dict) -> list[dict]:
    """Flatten every (summary_role, summary_field, sentence) triple across
    ALL roles -- this is what gets searched, globally, against ALL segments."""
    candidates = []
    for role, spec_or_list in field_specs.items():
        specs = spec_or_list if isinstance(spec_or_list, list) else [spec_or_list]
        for spec in specs:
            field = spec_label(spec)
            for sentence in extract_sentences_for_role(summary, spec):
                candidates.append({"summary_role": role, "summary_field": field, "summary_sentence": sentence})
    return candidates


def align_judgment(segments: list[dict], summary: dict, field_specs: dict, model_name: str,
                    threshold: float, bm25_min_score: float, rrf_k: int, role_bonus: float) -> list[dict]:
    if not segments:
        return []

    sentence_records = collect_candidate_sentences(summary, field_specs)
    if not sentence_records:
        return []

    # Precompute embeddings + BM25 index ONCE per judgment, over ALL segments
    # (global candidate pool -- no role restriction).
    seg_texts = [s["text"] for s in segments]
    seg_embs = embed(seg_texts, model_name)
    bm25 = build_bm25(seg_texts)

    sentences = [r["summary_sentence"] for r in sentence_records]
    sent_embs = embed(sentences, model_name)
    cosine_matrix = cosine_sim_matrix(sent_embs, seg_embs)  # (n_sentences, n_segments)

    # The maximum RRF score any candidate can possibly achieve is when it
    # ranks #1 on BOTH signals: 1/(k+1) + 1/(k+1) = 2/(k+1). role_bonus is a
    # FRACTION of that ceiling, not an absolute score -- this is what keeps
    # it a genuine tie-breaking nudge regardless of what rrf_k is set to.
    # (A previous version added role_bonus as a flat constant, e.g. 0.05,
    # against a real RRF range of ~0.0-0.033 -- that flat constant was
    # LARGER than the entire possible RRF spread, so it silently acted as a
    # near-absolute override instead of a nudge: every alignment came back
    # same-role, which is exactly the all-same-role behavior global search
    # was supposed to eliminate. Always sanity-check a "bonus" against the
    # actual scale of what it's being added to.)
    max_possible_rrf = 2.0 / (rrf_k + 1)
    role_bonus_abs = role_bonus * max_possible_rrf

    alignments = []
    for i, rec in enumerate(sentence_records):
        cosine_row = cosine_matrix[i]
        bm25_row = bm25_scores_for_query(bm25, rec["summary_sentence"])

        fused = rrf_fuse(cosine_row, bm25_row, k=rrf_k)
        # soft nudge: prefer segments whose own role matches the summary
        # sentence's expected role, WITHOUT excluding any other candidate
        role_match_mask = np.asarray([seg["role"] == rec["summary_role"] for seg in segments])
        fused_adjusted = fused + (role_bonus_abs * role_match_mask)

        # STRUCTURAL SAFEGUARD (belt-and-braces on top of the rescaling
        # above): the role bonus is only ever allowed to choose BETWEEN
        # candidates that already clear the acceptance gate on their own
        # merits -- it must never be the reason a summary sentence gets
        # rejected outright. Without this, even a well-calibrated bonus
        # could still flip an edge case sitting right at the gate boundary,
        # silently dropping an otherwise-valid pair.
        gate_pass = (cosine_row >= threshold) | (bm25_row >= bm25_min_score)
        if not gate_pass.any():
            continue  # no candidate clears the gate at all -- genuinely reject

        # Among gate-passing candidates only, let the role-boosted score
        # decide the winner; candidates that fail the gate are invisible to
        # this step no matter how the bonus would have ranked them.
        masked_scores = np.where(gate_pass, fused_adjusted, -np.inf)
        best_j = int(masked_scores.argmax())
        cosine_score = float(cosine_row[best_j])
        bm25_score = float(bm25_row[best_j])

        best_seg = segments[best_j]
        alignments.append({
            "summary_field": rec["summary_field"],
            "summary_role": rec["summary_role"],
            "summary_sentence": rec["summary_sentence"],
            "segment_id": best_seg["segment_id"],
            "segment_role": best_seg["role"],
            "segment_text": best_seg["text"],
            "role_match": bool(best_seg["role"] == rec["summary_role"]),
            "similarity_score": round(cosine_score, 4),   # kept for backward-compat
            "cosine_score": round(cosine_score, 4),
            "bm25_score": round(bm25_score, 4),
            "fused_score": round(float(fused_adjusted[best_j]), 6),
        })
    return alignments


def main(cfg: dict, limit: int | None, force: bool = False):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    seg_dir = Path(cfg["paths"]["stage0_dir"])
    out_dir = Path(cfg["paths"]["stage1_dir"])
    model_name = cfg["alignment"]["embedding_model"]
    threshold = cfg["alignment"]["threshold"]
    bm25_min_score = cfg["alignment"].get("bm25_min_score", 8.0)
    rrf_k = cfg["alignment"].get("rrf_k", 60)
    role_bonus = cfg["alignment"].get("role_bonus", 0.15)  # 0.15 = fraction-of-max-RRF default (see config.yaml)
    field_specs = cfg["summary_field_specs"]

    ids = iter_judgment_ids(raw_dir, limit=limit)
    print(f"Aligning {len(ids)} judgments (global search, cosine_threshold={threshold}, "
          f"bm25_min_score={bm25_min_score}, rrf_k={rrf_k}, role_bonus={role_bonus}, model={model_name})")

    all_for_audit = []
    total_aligned = 0
    total_role_matches = 0
    skipped = 0
    failed_judgments = []
    for jid in ids:
        seg_path = seg_dir / f"{jid}_segments.json"
        summary_path = raw_dir / f"{jid}_corrected_summary.json"
        out_path = out_dir / f"{jid}_alignments.json"
        if not seg_path.exists() or not summary_path.exists():
            continue

        try:
            if out_path.exists() and not force:
                # Resumability: embedding + BM25 computation is local (no API
                # cost) but not free time-wise at scale -- skip already-done
                # judgments by default. Still fold their existing alignments
                # into the audit sample / summary stats below so a resumed run
                # reports honest totals, not just the newly-processed subset.
                skipped += 1
                alignments = load_json(out_path)
            else:
                segments = load_json(seg_path)
                summary = load_gold_summary(summary_path)
                alignments = align_judgment(segments, summary, field_specs, model_name,
                                             threshold, bm25_min_score, rrf_k, role_bonus)
                for a in alignments:
                    a["judgment_id"] = jid
                save_json(out_path, alignments)
        except Exception as exc:
            # Judgment-level safety net, same pattern as Stage 2: one
            # malformed judgment (bad encoding, unexpected schema, etc.)
            # shouldn't lose progress on every other judgment in the batch.
            failed_judgments.append((jid, f"{type(exc).__name__}: {exc}"))
            continue

        all_for_audit.extend(alignments)
        total_aligned += len(alignments)
        total_role_matches += sum(1 for a in alignments if a["role_match"])

    sample_size = min(100, len(all_for_audit))
    audit_sample = random.sample(all_for_audit, sample_size) if all_for_audit else []
    save_json(out_dir / "_audit_sample.json", audit_sample)

    print(f"Done. {total_aligned} total alignments across {len(ids)} judgments -> {out_dir}")
    if skipped:
        print(f"  {skipped} judgment(s) skipped (already processed -- rerun with --force to redo).")
    if failed_judgments:
        print(f"  {len(failed_judgments)} judgment(s) FAILED and were skipped -- rerun this stage "
              f"(same command) to retry just these:")
        for jid, reason in failed_judgments:
            print(f"    - {jid}: {reason}")
    if total_aligned:
        pct = 100 * total_role_matches / total_aligned
        print(f"  {total_role_matches}/{total_aligned} ({pct:.1f}%) matched a segment of the 'expected' same role "
              f"-- the rest are cross-role matches the old same-role restriction would have missed entirely.")
    print(f"Audit sample of {sample_size} saved to {out_dir / '_audit_sample.json'} — inspect before scaling.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="reprocess judgments even if output already exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.limit, force=args.force)
