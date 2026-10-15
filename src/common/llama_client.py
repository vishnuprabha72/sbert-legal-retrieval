"""
Wrapper for calling Llama (via IBM watsonx.ai) to generate query variants
from an aligned (summary_sentence, segment_text) pair.

Requires environment variables:
    WATSONX_API_KEY
    WATSONX_PROJECT_ID
    WATSONX_URL          (e.g. https://us-south.ml.cloud.ibm.com)

Install: uv add ibm-watsonx-ai
"""
import json
import os
import re
import threading
import time
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()  # reads .env from the project root and populates os.environ

import httpx
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Check the watsonx.ai model catalog for the exact model id available in your
# project/region. Examples as of recent catalogs:
#   "meta-llama/llama-3-3-70b-instruct"
#   "meta-llama/llama-3-405b-instruct"
# NOTE: this is only a fallback default -- the actual model id used in the
# pipeline comes from config.yaml's query_generation.llama_model. Keep both
# in sync; a mismatch here has previously caused confusing "wrong model in
# use" debugging (config.yaml is the source of truth).
DEFAULT_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"

# Requests that legitimately fail transiently (slow model response, brief
# network blip) -- these are RETRIED with backoff, separate from the
# JSON-validation retry logic in stage2, which handles a different failure
# mode (the model responded fine but the content was malformed/wrong length).
_TRANSIENT_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout,
                          httpx.ConnectTimeout, ConnectionError)
_NETWORK_MAX_RETRIES = 3
_NETWORK_BACKOFF_SECONDS = 5  # doubles each retry: 5s, 10s, 20s

# Running requests concurrently (see stage2's ThreadPoolExecutor) makes
# hitting watsonx's per-account rate limit far more likely than the old
# strictly-sequential version ever was. IBM's SDK doesn't raise a single
# well-documented exception type for HTTP 429 across versions, so this
# checks the exception's message text instead -- broader than an exception
# type match, but deliberately narrow to just rate-limit wording so we don't
# accidentally swallow-and-retry genuine errors (bad model id, auth, etc.)
# forever.
_RATE_LIMIT_MARKERS = ("429", "too many requests", "rate limit", "throttl")


def _looks_like_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)

GEN_PARAMS = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.TEMPERATURE: 0.7,
    GenParams.REPETITION_PENALTY: 1.05,
}


@lru_cache(maxsize=4)
def _get_model(model_id: str) -> ModelInference:
    # NOTE on thread-safety: this is shared across worker threads when
    # stage2 runs with concurrency > 1. httpx's sync Client (which
    # ibm-watsonx-ai uses internally) is documented as safe for concurrent
    # use from multiple threads -- it manages its own connection pool.
    creds = Credentials(
        url=os.environ["WATSONX_URL"],
        api_key=os.environ["WATSONX_API_KEY"],
    )
    return ModelInference(
        model_id=model_id,
        credentials=creds,
        project_id=os.environ["WATSONX_PROJECT_ID"],
        params=GEN_PARAMS,
    )


PROMPT_TEMPLATE = """Below are two texts. Text A is a short human-verified summary point about this case.
Text B is the original paragraph from the actual court judgment that Text A summarizes.

TEXT A (LLM/human-generated case summary point):
"{summary_sentence}"

TEXT B (original verbatim judgment paragraph):
"{segment_text}"

THIS CASE'S OWN PARTIES (never use these to anchor any query -- see rule below): {case_parties}

Generate 3 search queries a lawyer might type to find TEXT B in a case-law search engine,
built around the legal CONCEPT, principle, or procedural point Text B is actually about.
Follow the word limits strictly.

CRITICAL RULE -- keep every query ABSTRACT and LEGAL, not a list of this case's own facts.
NEVER include, in any of the 3 queries:
  - this case's own party names: {case_parties}
  - vehicle registration / license plate numbers (e.g. "RJ 02 SA 7811")
  - exact calendar dates specific to this case (e.g. "October 2 2006", "30.03.2018")
  - specific monetary figures unique to this case (e.g. "Rs.17.69 crores")
  - names of victims/deceased/witnesses specific to this case
A lawyer searching by legal topic does not already know these one-off facts -- that's what
they're trying to find. A query built from them teaches "memorize this case's trivia," not
"retrieve by legal concept," which defeats the retrieval model's whole purpose.
INSTEAD, anchor on: the legal test/principle involved, the statute/section name and number
(these ARE reusable across similar cases, e.g. "Section 163A Motor Vehicles Act"), the
type of claim/proceeding, and the general fact PATTERN in the abstract (e.g. "fatal
motorcycle accident compensation claim", not the specific plate number or date of this one).
The ONLY exception: if Text B cites a DIFFERENT, separate precedent case (not this case's own
parties), that precedent's name is genuine citation search behavior and is fine to use, e.g.
"Associated Hotels v. Sardar Ranjit Singh".

IMPORTANT -- Text B may be a long paragraph shared by several different summary points
(e.g. a single paragraph discussing multiple cited precedents). Anchor every query,
especially small_query, on what is SPECIFIC to Text A's LEGAL content (the cited precedent
name, the statute section, the legal test) -- not a generic topic word that could equally
describe every other summary point drawn from the same paragraph, and not this case's own
facts as described above. If small_query would otherwise just be a generic phrase like
"subletting case law", prefer the specific legal term or cited-precedent name instead.

1. small_query: 5-8 words, keyword/search-bar style, no full sentence needed
2. medium_query: 8-20 words, a natural one-line question
3. large_query: 15-40 words, a detailed question including specific LEGAL detail
   (section numbers, statute names, the legal test/principle) -- still no case-specific
   facts (dates, plate numbers, amounts, party/victim names) per the rule above

Respond only in JSON, no other text before or after:
{{"small_query": "...", "medium_query": "...", "large_query": "..."}}
"""


LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "logs")
_FAILURE_LOG = os.path.join(LOG_DIR, "llama_generation_failures.log")
_LOG_LOCK = threading.Lock()  # multiple worker threads (stage2 concurrency) write to this file


def _log_failure(context: str, raw: str) -> None:
    """Every time we can't parse a JSON response, log the raw text instead of
    silently swallowing it -- this is what previously made empty query
    fields (word_count: 0) hard to diagnose: the pipeline had no record of
    *why* a field ended up blank."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with _LOG_LOCK:
            with open(_FAILURE_LOG, "a", encoding="utf-8") as f:
                f.write(f"--- {context} ---\n{raw!r}\n\n")
    except OSError:
        pass  # logging must never crash the pipeline


def _extract_json(text: str, context: str = "") -> dict | None:
    if not text:
        _log_failure(context or "empty_response", text or "")
        return None
    # 1) try a direct parse first -- covers the common well-behaved case
    #    without regex risk.
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 2) fall back to a NON-greedy brace match. The previous greedy
    #    `\{.*\}` regex could span across multiple JSON-looking fragments if
    #    the model emitted any stray text/braces, corrupting the parse. Our
    #    schema here is always flat (no nested objects), so a non-greedy
    #    match to the FIRST closing brace is safe and much more robust.
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    _log_failure(context or "unparseable_response", text)
    return None


def _call_watsonx(prompt: str, model_id: str) -> str | None:
    """Returns the raw text completion, or None if the request failed even
    after retrying transient network errors. Returning None (rather than
    raising) lets the caller fall through to the same "no valid response"
    handling already used for malformed JSON -- a slow/flaky network call
    degrades to a regeneration attempt or template fallback instead of
    crashing the entire multi-judgment run and losing all prior progress."""
    model = _get_model(model_id)
    last_exc = None
    for attempt in range(_NETWORK_MAX_RETRIES + 1):
        try:
            return model.generate_text(prompt=prompt)
        except _TRANSIENT_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < _NETWORK_MAX_RETRIES:
                wait = _NETWORK_BACKOFF_SECONDS * (2 ** attempt)
                _log_failure(f"network_retry (attempt {attempt + 1}/{_NETWORK_MAX_RETRIES}, "
                             f"waiting {wait}s): {type(exc).__name__}: {exc}", "")
                time.sleep(wait)
        except Exception as exc:
            # Rate-limit errors surface as whatever exception type the SDK
            # happens to wrap an HTTP 429 in -- checked by message content
            # rather than type (see _looks_like_rate_limit). Anything else
            # is a genuine error (bad model id, auth, etc.) and should
            # propagate immediately rather than retrying pointlessly.
            if _looks_like_rate_limit(exc) and attempt < _NETWORK_MAX_RETRIES:
                last_exc = exc
                wait = _NETWORK_BACKOFF_SECONDS * (2 ** attempt) * 2  # rate limits need more time to clear
                _log_failure(f"rate_limit_retry (attempt {attempt + 1}/{_NETWORK_MAX_RETRIES}, "
                             f"waiting {wait}s): {exc}", "")
                time.sleep(wait)
            else:
                raise
    _log_failure(f"network_exhausted: {type(last_exc).__name__}: {last_exc}", "")
    return None


def _format_prompt(summary_sentence: str, segment_text: str, case_parties: list[str] | None) -> str:
    parties_str = ", ".join(case_parties) if case_parties else "(not provided)"
    return PROMPT_TEMPLATE.format(summary_sentence=summary_sentence, segment_text=segment_text,
                                   case_parties=parties_str)


def generate_queries(summary_sentence: str, segment_text: str, model: str = DEFAULT_MODEL_ID,
                      case_parties: list[str] | None = None) -> dict | None:
    prompt = _format_prompt(summary_sentence, segment_text, case_parties)
    raw = _call_watsonx(prompt, model)
    return _extract_json(raw, context=f"generate_queries: {summary_sentence[:60]!r}")


def regenerate_field(field_name: str, prev_text: str, lo: int, hi: int,
                      summary_sentence: str, segment_text: str, model: str = DEFAULT_MODEL_ID,
                      had_previous_text: bool = True, case_parties: list[str] | None = None) -> str | None:
    """Corrective re-prompt for a single field that failed length validation
    (or was never produced at all)."""
    if had_previous_text and prev_text:
        correction_note = (
            f'Your previous answer for "{field_name}" was "{prev_text}" '
            f"({len(prev_text.split())} words). It must be between {lo}-{hi} words."
        )
    else:
        # IMPORTANT: don't claim there was a previous answer of "" -- that
        # previously confused the model into repeating empty output. If the
        # first call produced no parseable JSON at all, ask fresh instead.
        correction_note = f'Provide the "{field_name}" field, between {lo}-{hi} words.'

    correction_prompt = _format_prompt(summary_sentence, segment_text, case_parties) + (
        f"\n\n{correction_note} "
        f'Respond only in JSON with the single corrected field, no other text: '
        f'{{"{field_name}": "..."}}'
    )
    raw = _call_watsonx(correction_prompt, model)
    parsed = _extract_json(raw, context=f"regenerate_field[{field_name}]: {summary_sentence[:60]!r}")
    return parsed.get(field_name) if parsed else None


_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "was",
    "that", "this", "with", "as", "by", "at", "be", "it", "its", "his", "her",
    "which", "who", "whom", "under", "has", "have", "had", "not", "no",
}


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


def fallback_query(summary_sentence: str, lo: int, hi: int) -> str:
    """Last-resort, non-empty query when Llama generation fails validation
    even after retries. Extracts the first `hi` non-stopword-heavy tokens
    from the summary sentence itself -- guarantees SOME on-topic query text
    makes it into the training table instead of an empty string (which
    silently corrupts Stage 3's row count / dual-positive balance).

    Strips vehicle plates / exact dates / money figures from the source
    sentence BEFORE extracting tokens -- otherwise the fallback could
    reintroduce the exact case-specific-detail problem it exists to avoid,
    since summary sentences themselves often state these one-off facts."""
    cleaned = _VEHICLE_PLATE_RE.sub(" ", summary_sentence)
    cleaned = _NUMERIC_DATE_RE.sub(" ", cleaned)
    cleaned = _MONTHNAME_DATE_RE.sub(" ", cleaned)
    cleaned = _MONEY_RE.sub(" ", cleaned)
    words = re.findall(r"[A-Za-z0-9().,'-]+", cleaned)
    kept = [w for w in words if w.lower().strip(".,()") not in _STOPWORDS] or words
    kept = kept[:hi] if len(kept) > hi else kept
    if len(kept) < lo:
        # not enough words survived stripping -- pad back in from the
        # ORIGINAL sentence's stopword-filtered words (not the stripped
        # version) rather than falling short of the minimum, since a
        # too-short query is also invalid; this is a rare edge case for
        # very short summary sentences that were mostly case-specific detail
        orig_words = re.findall(r"[A-Za-z0-9().,'-]+", summary_sentence)
        kept = orig_words[:lo] if len(orig_words) >= lo else orig_words
    return " ".join(kept)
