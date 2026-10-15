"""
Stage 2 — For each aligned (summary_sentence, segment_text) pair, generate
3 query variants (small/medium/large) via Llama, with word-limit validation
and single-field regeneration on failure.

Input:  data/stage1_alignments/{id}_alignments.json
        data/raw/{id}_corrected_summary.json   (for this case's own party names)
Output: data/stage2_queries/{id}_queries.json

Requires IBM watsonx.ai credentials set as environment variables
(WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL) — see common/llama_client.py.

Run:
    uv run python src/stage2_llama_generate.py --limit 3
    uv run python src/stage2_llama_generate.py --limit 3 --force   # regenerate even if output exists
    uv run python src/stage2_llama_generate.py --workers 5         # process 5 judgments concurrently
                                                                     # (start conservative -- check your
                                                                     # watsonx plan's rate limit first)
"""
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common.io_utils import load_config, load_json, load_gold_summary, save_json, iter_judgment_ids
from common.llama_client import generate_queries, regenerate_field, fallback_query


def word_count(text: str) -> int:
    return len(text.split())


def extract_case_parties(summary: dict) -> list[str]:
    """Pull this case's own party names out of the gold summary so the
    prompt can explicitly tell the model NOT to anchor queries on them (a
    lawyer searching by topic doesn't already know the parties -- that's
    what they're trying to find). Also used for a belt-and-braces post-hoc
    check in case the model ignores the instruction."""
    parties = summary.get("Parties", {})
    if not isinstance(parties, dict):
        return []
    names = []
    for v in parties.values():
        if isinstance(v, str) and v.strip():
            names.append(v.strip())
    return names


# Case-specific-detail patterns: one-off facts unique to THIS case that a
# lawyer searching by legal topic wouldn't already know. A query built
# around these teaches "memorize this case's trivia," not "retrieve by
# legal concept" -- the opposite of what the retriever should learn.
_VEHICLE_PLATE_RE = re.compile(r"\b[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z0-9]{1,3}[\s-]?\d{3,4}\b")
_NUMERIC_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
_MONTHNAME_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(st|nd|rd|th)?,?\s*\d{4}\b|"
    r"\b\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|"
    r"November|December),?\s*\d{4}\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\bRs\.?\s?\d[\d,.]*\s?(crores?|lakhs?)?\b", re.IGNORECASE)


def _contains_case_party(text: str, case_parties: list[str]) -> bool:
    """True if the query text is substantially built around this case's own
    party name(s) rather than a different cited precedent. Matches on the
    first significant token of each party name (surnames/distinctive words)
    to catch partial forms like "Ambalal" appearing alone, not just the
    full "Ambalal Sarabhai Enterprise Limited"."""
    if not case_parties or not text:
        return False
    text_lower = text.lower()
    for party in case_parties:
        # take distinctive tokens (skip very short/common words like "Ltd", "LLP", "and")
        tokens = [t for t in re.findall(r"[A-Za-z]+", party) if len(t) > 3]
        if any(tok.lower() in text_lower for tok in tokens):
            return True
    return False


def _contains_case_specific_detail(text: str) -> bool:
    """Catches the broader category beyond party names: vehicle plates,
    exact calendar dates, and specific monetary figures -- all one-off case
    trivia, not reusable legal knowledge. Applied to ALL THREE query levels
    (small/medium/large), since even large_query should stay anchored on
    legal detail (section numbers, statute names, legal tests), not this
    case's own facts."""
    if not text:
        return False
    return bool(_VEHICLE_PLATE_RE.search(text) or _NUMERIC_DATE_RE.search(text)
                or _MONTHNAME_DATE_RE.search(text) or _MONEY_RE.search(text))


def validate_and_fix(parsed: dict, limits: dict, alignment: dict, model_name: str, max_retries: int,
                      seen_texts: dict[str, set[str]] | None = None,
                      case_parties: list[str] | None = None) -> dict:
    """
    seen_texts: optional {field_name: set(already-used query strings)} for the
    current judgment. When provided, a generated query that exactly repeats
    a previous one for the same field (common when several summary sentences
    share the same segment_text, e.g. multiple precedents inside one LAW
    paragraph -- the model tends to fall back to a generic phrase drawn from
    the shared segment instead of the sentence-specific detail) triggers a
    regeneration nudged toward the distinguishing content in this specific
    summary_sentence, up to max_retries.

    case_parties: this case's own party names. small_query/medium_query
    built mostly around these get flagged and regenerated -- a lawyer
    searching by legal topic doesn't already know the parties involved,
    so training the retriever on "party name -> paragraph" teaches a
    trivial shortcut instead of real topical/legal retrieval.
    """
    had_initial_json = parsed is not None
    result = {}
    # Party/case-specific-detail checks now apply to ALL THREE fields
    # uniformly. An earlier version exempted large_query from the party
    # check (reasoning: "more context, lower risk") -- a real example
    # proved that wrong: a large_query citing this case's own deceased
    # party by name ("...as in the case of Chotelal alias Shivram...")
    # slipped through untouched. The prompt already promises this rule
    # for all 3 queries; the code now actually enforces it everywhere too.

    for field, (lo, hi) in limits.items():
        text = parsed.get(field, "") if parsed else ""
        wc = word_count(text)
        valid = lo <= wc <= hi
        is_dup = bool(seen_texts is not None and text and text in seen_texts.get(field, set()))
        is_party_heavy = _contains_case_party(text, case_parties)
        is_case_specific = _contains_case_specific_detail(text)

        retries = 0
        while (not valid or is_dup or is_party_heavy or is_case_specific) and retries < max_retries:
            fixed = regenerate_field(
                field, text, lo, hi,
                alignment["summary_sentence"], alignment["segment_text"], model_name,
                had_previous_text=(had_initial_json and bool(text)),
                case_parties=case_parties,
            )
            if fixed:
                text = fixed
                wc = word_count(text)
                valid = lo <= wc <= hi
                is_dup = bool(seen_texts is not None and text in seen_texts.get(field, set()))
                is_party_heavy = _contains_case_party(text, case_parties)
                is_case_specific = _contains_case_specific_detail(text)
            retries += 1

        used_fallback = False
        fallback_reason = None
        if not valid or is_party_heavy or is_case_specific:
            # Record WHY, not just THAT, a fallback happened -- previously
            # only "used_fallback: true" was recorded with no indication of
            # cause, which made a question like "why did judgment 10 have
            # 13 fallbacks?" unanswerable without guessing (the shared
            # failure log only captures a DIFFERENT category: malformed
            # JSON / network issues -- these validation-level fallbacks
            # never appeared there at all).
            reasons = []
            if not valid:
                reasons.append("word_count_out_of_range")
            if is_party_heavy:
                reasons.append("case_party_name")
            if is_case_specific:
                reasons.append("case_specific_detail")  # vehicle plate / date / money figure
            if is_dup:
                reasons.append("duplicate_within_judgment")
            fallback_reason = ",".join(reasons)

            # Last resort: never leave a field empty/invalid/case-specific in
            # the final table -- fallback_query is built from the summary
            # sentence itself, which is far less likely to carry this case's
            # incidental trivia than a shared judgment paragraph is.
            text = fallback_query(alignment["summary_sentence"], lo, hi)
            wc = word_count(text)
            valid = lo <= wc <= hi
            used_fallback = True

        if seen_texts is not None:
            seen_texts.setdefault(field, set()).add(text)

        result[field] = {"text": text, "word_count": wc, "valid": valid, "used_fallback": used_fallback,
                          "fallback_reason": fallback_reason}
    return result


def _process_one_judgment(jid: str, align_dir: Path, raw_dir: Path, out_dir: Path,
                           model_name: str, limits: dict, max_retries: int) -> dict:
    """Everything needed to process ONE judgment, isolated so it can safely
    run in its own worker thread: seen_texts (dedup tracking) is created
    fresh here and never shared across threads, so no locking is needed for
    it -- each judgment's dedup is naturally confined to the thread handling
    that judgment."""
    align_path = align_dir / f"{jid}_alignments.json"
    alignments = load_json(align_path)
    summary_path = raw_dir / f"{jid}_corrected_summary.json"
    case_parties = extract_case_parties(load_gold_summary(summary_path)) if summary_path.exists() else []

    results = []
    seen_texts: dict[str, set[str]] = {}
    for a in alignments:
        parsed = generate_queries(a["summary_sentence"], a["segment_text"], model_name,
                                   case_parties=case_parties)
        validated = validate_and_fix(parsed, limits, a, model_name, max_retries,
                                      seen_texts=seen_texts, case_parties=case_parties)
        results.append({**a, "queries": validated})

    save_json(out_dir / f"{jid}_queries.json", results)

    return {
        "jid": jid,
        "pairs": len(alignments),
        "valid": sum(sum(1 for v in r["queries"].values() if v["valid"]) for r in results),
        "fallback": sum(sum(1 for v in r["queries"].values() if v.get("used_fallback")) for r in results),
    }


def main(cfg: dict, limit: int | None, force: bool = False, workers: int = 1):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    align_dir = Path(cfg["paths"]["stage1_dir"])
    out_dir = Path(cfg["paths"]["stage2_dir"])
    model_name = cfg["query_generation"]["llama_model"]
    limits = {k: tuple(v) for k, v in cfg["query_generation"]["word_limits"].items()}
    max_retries = cfg["query_generation"]["max_retries"]
    workers = workers or cfg["query_generation"].get("max_workers", 1)

    ids = iter_judgment_ids(raw_dir, limit=limit)
    todo = []
    skipped = 0
    for jid in ids:
        align_path = align_dir / f"{jid}_alignments.json"
        out_path = out_dir / f"{jid}_queries.json"
        if not align_path.exists():
            continue
        # RESUMABILITY: decided BEFORE scheduling any work, so an
        # already-completed judgment is never even submitted to a worker
        # thread -- zero wasted API calls on a resumed run.
        if out_path.exists() and not force:
            skipped += 1
            continue
        todo.append(jid)

    print(f"Generating queries for {len(todo)} judgment(s) using {model_name} "
          f"({workers} concurrent worker{'s' if workers != 1 else ''})")
    if workers > 1:
        print("  NOTE: check your watsonx plan's concurrent-request limit before raising --workers further; "
              "rate-limit errors are auto-retried with backoff, but too high a worker count will just "
              "spend most of its time waiting on 429s instead of going faster.")

    total_pairs = 0
    total_valid = 0
    total_fallback = 0
    failed_judgments = []
    print_lock = threading.Lock()
    done_count = 0

    def run_one(jid: str):
        try:
            return jid, _process_one_judgment(jid, align_dir, raw_dir, out_dir, model_name, limits, max_retries), None
        except Exception as exc:
            return jid, None, f"{type(exc).__name__}: {exc}"

    if workers <= 1:
        # Sequential path kept as a simple, dependency-free fallback --
        # identical behavior to concurrent path with workers=1, just
        # without any thread-pool machinery.
        results_iter = (run_one(jid) for jid in todo)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {executor.submit(run_one, jid): jid for jid in todo}
        results_iter = (f.result() for f in as_completed(futures))

    try:
        for jid, stats, error in results_iter:
            done_count += 1
            with print_lock:
                if error:
                    failed_judgments.append((jid, error))
                    print(f"  [{done_count}/{len(todo)}] {jid}: FAILED -- {error}")
                else:
                    total_pairs += stats["pairs"]
                    total_valid += stats["valid"]
                    total_fallback += stats["fallback"]
                    print(f"  [{done_count}/{len(todo)}] {jid}: {stats['pairs']} pairs, "
                          f"{stats['valid']} valid fields, {stats['fallback']} fallback")
    finally:
        if workers > 1:
            executor.shutdown(wait=True)

    print(f"\nDone. {total_pairs} aligned pairs processed, {total_valid} valid query fields generated -> {out_dir}")
    if skipped:
        print(f"  {skipped} judgment(s) skipped (already had output -- rerun with --force to regenerate).")
    if total_fallback:
        print(f"  {total_fallback} field(s) fell back to a template-extracted query after exhausting retries "
              f"(includes party-name-heavy queries that couldn't be fixed via regeneration) "
              f"-- check data/logs/llama_generation_failures.log for the raw model output on those.")
    if failed_judgments:
        print(f"  {len(failed_judgments)} judgment(s) FAILED and were skipped -- rerun this stage "
              f"(same command) to retry just these, already-completed judgments won't be re-run:")
        for jid, reason in failed_judgments:
            print(f"    - {jid}: {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                         help="Regenerate all judgments even if output already exists (default: skip completed ones)")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of judgments to process concurrently (default: config's "
                              "query_generation.max_workers, or 1 if unset). Start small (3-5) and increase "
                              "only if you don't see rate-limit retries in the log.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.limit, force=args.force, workers=args.workers)
