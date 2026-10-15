"""
controlled_eval_app.py

Paragraph-level evidence retrieval within a single judgment: BM25 against
sentence-embedding models, with an interactive mode and a batch evaluation
mode. The batch mode produced the controlled-evaluation results in Section 5.1
of the paper (Tables 1 and 2).

Run:
    uv run streamlit run evaluation/controlled_eval_app.py

Modes
-----
Single judgment: upload one judgment (PDF, TXT or a structured JSON summary),
type a query, and compare the top-ranked paragraph for every selected model
and inference encoding at a chosen BM25/semantic weight.

Batch evaluation: upload a query CSV (query_id, judgment_id, query_text,
gold_para_num; optionally query_style and query_length_class) and the
judgment files it refers to. For each query and each model/encoding pair, the
app records the rank of the gold paragraph under semantic-only scoring
(BM25 weight 0) and BM25-only scoring (weight 1), and reports Hit@k, MRR, a
per-query winner, and the query terms BM25 matched in the gold paragraph and
in its own top-1 paragraph. Every aggregate table reports coverage (how many
gold paragraphs were found in the parse), so accuracy is never computed over
a silently reduced set of queries.

Models: untuned all-MiniLM-L6-v2, Swiss Legal SBERT, InLegalBERT (mean pooling
over a 512-token window, no contrastive tuning), and our Legacy-trained and
CP-trained checkpoints. Each can be run with legacy (truncated) encoding or
chunk-pool mean encoding at inference, so the training regime and the
inference encoding can be varied independently. Under chunk-pool encoding a
paragraph longer than the model's token budget is split at sentence
boundaries, each chunk is embedded, and the chunk vectors are mean-pooled.
Note that chunk-pooled vectors are not re-normalised, which does not affect
rankings but does affect the absolute cosine floors used by the relevance
gate below.

Paragraph splitting
-------------------
JSON: every leaf field becomes one paragraph labelled with its field path.
TXT/PDF: paragraph boundaries are taken as the longest increasing sequence of
"N." markers in the text, which separates the judgment's own numbering from
numbers inside quoted passages and sub-lists. Each paragraph keeps its
original number, so batch mode matches gold_para_num to the paragraph with
that number rather than to a list position. For PDFs, running headers and
footers are removed first. Paragraphs are tagged with a rhetorical role by
rule-based patterns (for display only; roles play no part in scoring).

Known limitation: in footnote-heavy judgments the increasing sequence can
jump to footnote reference numbers and skip some real paragraphs. Check the
parsed paragraph list before querying; in batch mode such queries are
reported as "gold paragraph not found" and counted against coverage.

Relevance gate (single-judgment mode only)
------------------------------------------
Min-max normalisation always puts some paragraph at the top of the range,
even for an unrelated query. The top result is therefore labelled as
unconfirmed unless its raw cosine similarity or its raw BM25 score (with at
least two distinct query terms matched) clears a floor. A small gap between
the top two combined scores is also flagged. Both checks are informational
and do not change the ranking; the default floors are starting points and
should be calibrated on labelled queries.

Implementation notes
--------------------
The cached embedder is shared across Streamlit sessions, so encoding calls are
serialised behind a lock and TOKENIZERS_PARALLELISM is disabled. Embedding
caches are keyed on each local checkpoint's modification time, so a
retrained model saved to the same path is picked up. Model locations can be
set with LEGACY_MODEL_PATH, CHUNKPOOL_MODEL_PATH and CHUNK_POOL_SRC_PATH.

Requires: streamlit, nltk, pdfplumber (PDF input only); `uv sync --extra eval`.
"""

import os
# Must be set before any tokenizer is loaded (see the implementation notes
# in the module docstring).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
import io, re, json, sys, threading
import pandas as pd
import numpy as np
from collections import Counter
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

for _pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

_ENCODE_LOCK = threading.Lock()

# Model paths: set LEGACY_MODEL_PATH / CHUNKPOOL_MODEL_PATH to a local folder or a
# Hugging Face model id; defaults point inside this repository's data/models/.
EMBED_MODELS = {
    "MiniLM (general)":            "sentence-transformers/all-MiniLM-L6-v2",
    "Swiss Legal SBERT (domain)":  "Stern5497/sBert-swiss-legal-base",
    "InLegalBERT (Indian domain, no SBERT tuning)": "law-ai/InLegalBERT",
    "Mine_Finetuned_Indian_Legal": os.environ.get(
        "LEGACY_MODEL_PATH",
        str(REPO_ROOT / "data" / "models" / "sbert-legal-v1-legacy"),
    ),
    "Mine_Finetuned_Indian_Legal_ChunkPool": os.environ.get(
        "CHUNKPOOL_MODEL_PATH",
        str(REPO_ROOT / "data" / "models" / "sbert-legal-v1-chunkpoolmean"),
    ),
}
# InLegalBERT is a raw HF BERT checkpoint with no built-in SBERT pooling
# head, so it needs the manual Transformer+Pooling wrapper in get_embedder()
# below. Both Mine_Finetuned_* entries are self-contained sentence-
# transformers model folders (saved via .save()) and do NOT need this.
NEEDS_MANUAL_POOLING = {"law-ai/InLegalBERT"}
# InLegalBERT is a BERT-base checkpoint with 512 position embeddings, so it
# gets its full native window (the MiniLM-based models are limited to 256).
MANUAL_POOLING_MAX_SEQ_LEN = 512
MANUAL_POOLING_MODE = "mean"

WEIGHT_GRID = [(0.0,1.0), (0.3,0.7), (0.5,0.5), (0.7,0.3), (1.0,0.0)]

# ── Chunk-pool encoding (shared with training; see src/common/chunk_pool_encoding.py) ──
# Defaults to this repository's src/ folder; override with CHUNK_POOL_SRC_PATH.
CHUNK_POOL_SRC_PATH = os.environ.get(
    "CHUNK_POOL_SRC_PATH",
    str(REPO_ROOT / "src"),
)
if CHUNK_POOL_SRC_PATH not in sys.path:
    sys.path.insert(0, CHUNK_POOL_SRC_PATH)
from common.chunk_pool_encoding import encode_with_chunk_pool
import torch

# Inference encoding is independent of the model: any model, including ones
# not trained with chunk-and-pool, can be run with chunk-pool encoding. This
# separates the effect of the encoding from the effect of fine-tuning.
ENCODING_MODES = {
    "Legacy (truncated)": None,
    "Chunk-Pool Mean":    "mean",
}
CHUNK_POOL_BATCH_SIZE = 8  # small batches: long texts expand into many chunks

# ── Relevance-gate defaults (single-judgment mode only; not used in batch
# evaluation). Starting points, not calibrated values. They sit below the 0.6
# cosine threshold of Stage 1 because the best match within one judgment
# tends to score lower than the best match across a whole corpus.
#
# The cosine floor is set per model and can be adjusted per encoding mode in
# the UI. Models without contrastive training (InLegalBERT) give high cosine
# similarity to almost any pair of texts, hence the higher default; its 0.55
# is a rough guess and has not been calibrated. The BM25 floor is shared,
# since BM25 does not depend on the embedding model.
DEFAULT_COSINE_MIN_BY_MODEL = {
    "MiniLM (general)": 0.35,
    "Swiss Legal SBERT (domain)": 0.35,
    "InLegalBERT (Indian domain, no SBERT tuning)": 0.55,   # not calibrated, see above
    "Mine_Finetuned_Indian_Legal": 0.35,
    "Mine_Finetuned_Indian_Legal_ChunkPool": 0.35,
}
DEFAULT_COSINE_MIN_FALLBACK = 0.35     # used for any model label not listed above
DEFAULT_BM25_MIN_RAW_THRESHOLD = 3.0   # raw rank_bm25 Okapi score (pre-minmax) — shared across models
DEFAULT_SCORE_GAP_MIN = 0.03           # combined-score gap between rank 1 and rank 2
DEFAULT_MIN_TERM_OVERLAP = 2           # min distinct query terms that must appear in the paragraph
                                        # for the BM25 branch to count — guards against BM25's known
                                        # short-paragraph score inflation from a single coincidental word

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL LOADING (identical pattern to the AILA / paragraph scripts)
# ══════════════════════════════════════════════════════════════════════════════

def _load_pooling_classes():
    try:
        from sentence_transformers.sentence_transformer.modules import Transformer, Pooling
        return Transformer, Pooling
    except ImportError:
        from sentence_transformers.models import Transformer, Pooling
        return Transformer, Pooling

def _embedding_dim(word_embedding_model):
    if hasattr(word_embedding_model, "get_embedding_dimension"):
        return word_embedding_model.get_embedding_dimension()
    return word_embedding_model.get_word_embedding_dimension()

# Cache key for a model: the latest modification time of a local checkpoint
# folder, so a model retrained to the same path is reloaded rather than
# served from Streamlit's cache. Hub model ids get a constant key.
def _model_version_key(model_name):
    if os.path.isdir(model_name):
        candidates = [os.path.join(model_name, f) for f in ("config.json", "modules.json")]
        mtimes = [os.path.getmtime(p) for p in candidates if os.path.isfile(p)]
        try:
            return max(mtimes) if mtimes else os.path.getmtime(model_name)
        except OSError:
            return 0.0
    return 0.0

@st.cache_resource(show_spinner=False)
def get_embedder(model_name, _version_key=0.0):
    # NOTE: this object is cached and SHARED across every Streamlit session/
    # thread on the server for the lifetime of the process — callers must go
    # through _ENCODE_LOCK (see encode_texts below), not call
    # .encode()/.tokenize() on it directly. See the implementation notes in the module docstring.
    # `_version_key` is only part of the cache key (see _model_version_key).
    from sentence_transformers import SentenceTransformer
    if model_name in NEEDS_MANUAL_POOLING:
        Transformer, Pooling = _load_pooling_classes()
        word_embedding_model = Transformer(model_name, max_seq_length=MANUAL_POOLING_MAX_SEQ_LEN)
        pooling_model = Pooling(_embedding_dim(word_embedding_model), pooling_mode=MANUAL_POOLING_MODE)
        return SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
    return SentenceTransformer(model_name, device="cpu")

@torch.no_grad()
def encode_texts(model_name, texts, encoding_mode_label, legacy_batch_size=32):
    """
    Universal encoder used for BOTH paragraphs and queries. encoding_mode_
    label is a key into ENCODING_MODES ("Legacy (truncated)" or
    "Chunk-Pool Mean"). Any model can be run under either mode.

    Legacy branch: unchanged behavior — plain model.encode(), truncates at
    max_seq_length, L2-normalized.

    Chunk-Pool branch: splits each text into sentence-bounded, token-exact
    chunks (only if it exceeds the token budget — short paragraphs and
    virtually all queries resolve to a single chunk), embeds each chunk
    with gradients disabled (@torch.no_grad(), pure inference), and
    mean-pools the chunk embeddings into one vector per input text. See the
    module docstring's score-scale caveat re: normalization — this has NOT
    been verified to match model.encode(normalize_embeddings=True)'s scale.

    Runs under torch.no_grad() and behind _ENCODE_LOCK, because the cached
    model and tokenizer are shared across Streamlit sessions.
    """
    embedder = get_embedder(model_name, _model_version_key(model_name))
    texts = list(texts)
    pooling = ENCODING_MODES[encoding_mode_label]

    with _ENCODE_LOCK:
        if pooling is None:  # legacy -- unchanged behavior, truncates at max_seq_length
            return embedder.encode(
                texts, batch_size=legacy_batch_size, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype("float32")

        vecs = []
        for i in range(0, len(texts), CHUNK_POOL_BATCH_SIZE):
            batch = texts[i:i + CHUNK_POOL_BATCH_SIZE]
            vecs.append(encode_with_chunk_pool(embedder, batch, max_tokens=None, pooling=pooling).cpu())
        if vecs:
            return torch.cat(vecs, dim=0).numpy().astype("float32")
        return np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype="float32")

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]

def _minmax(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-9) if mx > mn else np.zeros_like(a)

# ══════════════════════════════════════════════════════════════════════════════
# PARAGRAPH SPLITTING — validated against real uploaded samples (see docstring)
# ══════════════════════════════════════════════════════════════════════════════

def split_json_judgment(data, prefix=""):
    """Recursive: every leaf string becomes one path-labeled paragraph."""
    paragraphs = []
    if isinstance(data, dict):
        for key, value in data.items():
            label = f"{prefix}{key}" if not prefix else f"{prefix} - {key}"
            if isinstance(value, str):
                if value.strip():
                    paragraphs.append(f"{label}: {value.strip()}")
            elif isinstance(value, (dict, list)):
                paragraphs.extend(split_json_judgment(value, label))
    elif isinstance(data, list):
        for i, item in enumerate(data, 1):
            item_label = f"{prefix} #{i}"
            if isinstance(item, str):
                if item.strip():
                    paragraphs.append(f"{item_label}: {item.strip()}")
            else:
                paragraphs.extend(split_json_judgment(item, item_label))
    return paragraphs

# ── TXT/PDF splitter: quote-aware, LIS-based real-paragraph detection +
# rule-based rhetorical role tagging (see module docstring) ──

def _nratio(a, b):
    """Bigram Jaccard similarity — used for header/footer detection."""
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    a, b = a.lower().strip(), b.lower().strip()
    def bg(s): return set(s[i:i+2] for i in range(len(s)-1))
    ba, bb = bg(a), bg(b)
    if not ba and not bb: return 1.0
    if not ba or not bb:  return 0.0
    return len(ba & bb) / len(ba | bb)

def _detect_header_footer(pre, n):
    idxs  = sorted(set([0, max(0, n-2), n//2, n//4, n//8]))
    sampl = [pre[i] for i in idxs]
    ns    = len(sampl)

    def _hcand(t):
        parts = t.split('\n\n', 1)
        lines = [l.strip() for l in parts[0].split('\n') if l.strip()]
        while lines and re.match(r'^\d+$', lines[0]):
            lines = lines[1:]
        return lines[0] if lines else ''

    def _fcand(t):
        rev = t.split('\n\n', 1)[-1][::-1]
        return rev.split('\n', 1)[0][::-1].strip()

    hc = [_hcand(p) for p in sampl]
    fc = [_fcand(p) for p in sampl]
    hs = [round(_nratio(hc[0], hc[j])) for j in range(ns)]
    fs = [round(_nratio(fc[0], fc[j])) for j in range(ns)]

    running_header = hc[0] if (hc[0] and hs == [1]*ns) else None
    running_footer = fc[0] if (fc[0] and fs == [1]*ns) else None

    def _finv(f):
        return re.sub(r'\s+[Pp]age\s+\d+\s+of\s+\d+\s*$', '', f).strip()
    fc_inv = [_finv(f) for f in fc]
    fi_s   = [round(_nratio(fc_inv[0], fc_inv[j])) for j in range(ns)]
    paged_pfx = (fc_inv[0]
                 if (fc_inv[0] and not running_footer
                     and fi_s == [1]*ns and fc_inv[0] != fc[0])
                 else None)

    _pgnum = re.compile(r'^\d+(\s*[\|/]\s*[\w\s]+)?$')
    has_pg_ftr = (not running_footer and not paged_pfx
                  and all(_pgnum.match(f) for f in fc if f))

    return running_header, running_footer, paged_pfx, has_pg_ftr, _pgnum

def _find_para_boundary(text):
    """Locate where numbered paragraphs truly begin, past the title block."""
    pats = [
        re.compile(r'(?<=\s)(?!\d{4}[. ])\d{1,3}\. (?=[A-Z](?:[a-z]|\s[a-z]))'),
        re.compile(r'(?<=\s)(?<!\()(?<!\d)\d{1,3}\) (?=[A-Z])'),
        re.compile(r'(?<=\s)(?!\d{4}\b)(\d{1,3}) (?=[A-Z](?:[a-z]{2,}|\s[a-z]{2,}))'),
    ]

    def _n(m, t):
        return int(re.match(r'\d+', t[m.start():]).group())

    for pat in pats:
        for m in pat.finditer(text):
            n = _n(m, text)
            if n > 10:
                continue
            window   = text[m.start(): m.start() + 8000]
            seq_nums = [_n(mm, window) for mm in pat.finditer(window)]
            if (n + 1) in seq_nums:
                return m.start()

    for pat in pats:
        for m in pat.finditer(text):
            if _n(m, text) <= 5:
                return m.start()

    return None

def _find_all_candidate_splits(text):
    """All positions where a paragraph-number pattern occurs (real or quoted)."""
    pats = [
        re.compile(r'(?:(?<=\s)|(?<=^))(?!\d{4})(\d{1,3})\.(\d{1,3})\.?\s+(?=[A-Z](?:[a-z]|\s[a-z]))'),
        re.compile(r'(?:(?<=\s)|(?<=^))(?!\d{4}[. ])(\d{1,3})\.\s+(?=[A-Z](?:[a-z]|\s[a-z]))'),
        re.compile(
            r'(?:(?<=[.!?]\s)|(?<=["\')\u201d\u2019\u2018\u201c]\s))'
            r'(?!\d{4}\b)(\d{1,3})\s+(?=[A-Z](?:[a-z]{2,}|\s[a-z]{2,}))'
        ),
    ]

    candidates = []
    for pat in pats:
        for m in pat.finditer(text):
            if pat.groups >= 2 and m.lastindex and m.lastindex >= 2:
                major = int(m.group(1)); minor = int(m.group(2)); label = f"{major}.{minor}"
            else:
                major = int(m.group(1)); minor = 0; label = str(major)
            candidates.append((m.start(), m.end(), major, minor, label))

    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    unique = []
    last_end = -1
    for c in candidates:
        if c[0] >= last_end:
            unique.append(c); last_end = c[1]

    _list_intro = re.compile(r'[:\-]{1,2}\s*(?:"[^"]*")?\s*(?:[A-Z][A-Z ]+)?\s*$')
    filtered = []
    for c in unique:
        pre = text[max(0, c[0] - 250): c[0]]
        if c[2] <= 20 and _list_intro.search(pre.rstrip()):
            continue
        filtered.append(c)
    return filtered

def _find_best_paragraph_sequence(candidates):
    """Longest strictly-increasing-by-number subsequence — the real judgment paragraphs."""
    if not candidates:
        return []
    n = len(candidates)
    def _key(c): return c[2] * 10000 + c[3]
    keys = [_key(c) for c in candidates]

    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if keys[j] < keys[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1; parent[i] = j

    best_len = max(dp)
    best_end, best_start_key = -1, float('inf')
    for i in range(n):
        if dp[i] == best_len:
            j = i
            while parent[j] != -1:
                j = parent[j]
            if keys[j] < best_start_key:
                best_start_key = keys[j]; best_end = i

    seq_indices = []
    j = best_end
    while j != -1:
        seq_indices.append(j); j = parent[j]
    seq_indices.reverse()
    return [candidates[i] for i in seq_indices]

def _split_paragraphs_core(text):
    """Split judgment body into numbered paragraphs via the LIS best-path."""
    candidates = _find_all_candidate_splits(text)
    if not candidates:
        return [{'number': 0, 'label': 'intro', 'text': text.strip()}]

    best_seq = _find_best_paragraph_sequence(candidates)
    if not best_seq:
        return [{'number': 0, 'label': 'intro', 'text': text.strip()}]

    paragraphs = []
    first_pos = best_seq[0][0]
    if first_pos > 0:
        pre_text = text[:first_pos].strip()
        if pre_text:
            paragraphs.append({'number': 0, 'label': 'intro', 'text': pre_text})

    for i, (start, end, major, minor, label) in enumerate(best_seq):
        next_start = best_seq[i + 1][0] if i + 1 < len(best_seq) else len(text)
        para_text = text[start:next_start].strip()
        paragraphs.append({'number': major, 'label': label, 'text': para_text})
    return paragraphs

_ROLE_RULES = [
    ('ORDER', [
        r'\bappeal\s+is\s+(hereby\s+)?(dismissed|allowed|disposed)',
        r'\bwe\s+(hereby\s+)?direct\b',
        r'\baccordingly.{0,30}(dismissed|allowed|disposed)',
        r'\bno\s+order\s+as\s+to\s+costs\b',
        r'\bbail\s+bonds?\s+are\s+cancelled\b',
        r'\bshall\s+surrender\b',
        r'\binterim\s+order.{0,20}vacated\b',
        r'\bthe\s+(above|present)\s+appeal.{0,40}(fail|dismiss|allow)',
        r'\bpending\s+applications?.{0,30}disposed\b',
    ]),
    ('HOLDING', [
        r'\bwe\s+(are\s+of\s+the\s+(view|opinion)|hold|conclude|find)\b',
        r'\bwe\s+are\s+(clearly|therefore|thus)\s+of\s+the\s+(view|opinion)\b',
        r'\bin\s+(our|the)\s+(considered\s+)?view\b',
        r'\bfor\s+(the\s+)?foregoing\s+reasons\b',
        r'\bfor\s+what\s+has\s+been\s+(discussed|noted|observed)\b',
        r'\bthe\s+answer\s+to\s+(the\s+)?question\b',
        r'\baccordingly.{0,40}(we|this court)\b',
        r'\bwe\s+therefore\s+(hold|conclude|find|are)\b',
    ]),
    ('ISSUE', [
        r'\bthe\s+(main\s+|only\s+|sole\s+|primary\s+|central\s+)?question.{0,60}(is|arises|for\s+determination)\b',
        r'\bquestion\s+(of\s+law\s+)?for\s+(our\s+)?consideration\b',
        r'\bthe\s+(only\s+|main\s+)?issue.{0,30}(before|arises|is)\b',
        r'\bpoints?\s+(arising\s+)?for\s+determination\b',
        r'\bthe\s+question\s+that\s+(falls|arises|needs)\b',
    ]),
    ('ARG_P', [
        r'\b(learned\s+)?(senior\s+)?counsel\s+(for\s+the\s+)?(appellant|petitioner|accused)\b',
        r'\b(appellant|petitioner)\s+(has\s+|have\s+)?(submitted|contended|argued|urged)\b',
        r'\bon\s+behalf\s+of\s+the\s+(appellant|petitioner|accused)\b',
        r'\blearned\s+(senior\s+)?counsel.{0,40}(for\s+)?appellant\b',
        r'\bsubmission.{0,30}(on\s+behalf\s+of\s+)?(the\s+)?(appellant|petitioner)\b',
        r'\bthe\s+appellant.{0,30}(submits?|contends?|urges?|argues?|pleads?)\b',
        r'\bit\s+(is|was)\s+(submitted|contended|argued|urged)\s+(by|on\s+behalf\s+of)\s+(the\s+)?(appellant|petitioner|accused)\b',
        r'\bthe\s+(first|main|primary|sole)\s+contention\s+of\s+(the\s+)?(appellant|petitioner)\b',
        r'\b(mr|ms|mrs|dr|shri|smt)\.?\s+\w+.{0,40}(for\s+the\s+)?(appellant|petitioner)\b',
        r'\blearned\s+(senior\s+)?advocate.{0,40}(appellant|petitioner)\b',
        r'\bappellant.s?\s+(counsel|advocate|contention|submission)\b',
        r'\bpetitioner.s?\s+(counsel|advocate|contention|submission)\b',
        r'\bit\s+is\s+(further\s+)?submitted\s+(that|by)\b',
        r'\bthe\s+accused.{0,30}(submits?|contends?|argues?|denies?)\b',
    ]),
    ('ARG_R', [
        r'\b(learned\s+)?(senior\s+)?counsel\s+(for\s+the\s+)?respondent\b',
        r'\brespondent.{0,30}(submitted|contended|argued|urged)\b',
        r'\bon\s+behalf\s+of\s+the\s+respondent\b',
        r'\blearned\s+(senior\s+)?counsel.{0,40}(for\s+)?respondent\b',
        r'\bthe\s+state.{0,30}(submits?|contends?|argues?|urges?)\b',
        r'\blearned\s+(government\s+)?pleader\b',
        r'\blearned\s+(public\s+)?prosecutor\b',
        r'\bit\s+(is|was)\s+(submitted|contended|argued)\s+(by|on\s+behalf\s+of)\s+(the\s+)?respondent\b',
        r'\bthe\s+respondent.{0,50}(submits?|contends?|urges?|argues?|pleads?)\b',
        r'\b(mr|ms|mrs|dr|shri|smt)\.?\s+\w+.{0,40}for\s+the\s+respondent\b',
        r'\blearned\s+(senior\s+)?advocate.{0,40}respondent\b',
        r'\brespondent.s?\s+(counsel|advocate|contention|submission)\b',
        r'\bthe\s+(union\s+of\s+india|government).{0,30}(submits?|contends?|argues?)\b',
        r'\bthe\s+counter.?affidavit\b',
        r'\bthe\s+reply\s+(filed|submitted)\s+by\s+(the\s+)?respondent\b',
    ]),
    ('LAW', [
        r'\bsection\s+\d+\s+of\s+the\b',
        r'\barticle\s+\d+\s+of\s+the\b',
        r'\bthe\s+\w[\w\s]+act[,\s]+\d{4}\b',
        # Brackets around the year are optional: extracted text often has
        # "(1964 7 SCR 123)" rather than "(1964) 7 SCR 123".
        r'\(?\d{4}\)?\s+\d+\s+(SCC|SCR|AIR|All|Bom|Mad|Cal)\b',
        r'\breported\s+(as|in)\b',
        # "AIR1956" without a space is common in extracted PDF text.
        r'\bAIR\s?\d{4}\b',
        r'\bthis\s+court\s+(has\s+)?held\s+in\b',
        r'\bin\s+the\s+case\s+of\b',
        r'\breliance\s+(is\s+)?placed\s+on\b',
        r'\bthe\s+law\s+(laid\s+down|settled)\b',
        r'\bprecedent\b',
        r'\bratio\s+decidendi\b',
        # Paragraphs that quote a statute verbatim, e.g.
        # "Sections 7 and 9 are as under:- '7. Functions of...'".
        r'\b(sections?|articles?|clauses?)\s+\d+.{0,40}\bas\s+(under|follows)\b',
        r'\bnotwithstanding\s+anything\s+contained\s+in\s+any\s+other\s+law\b',
    ]),
    ('REASON', [
        r'\bhaving\s+(regard|considered|heard)\b',
        r'\bwe\s+have\s+(considered|perused|examined|gone through)\b',
        r'\bin\s+(our\s+)?(view|opinion|considered opinion)\b',
        r'\bwe\s+(note|observe|find|notice)\s+that\b',
        r'\bit\s+is\s+(clear|evident|apparent|manifest|obvious)\s+(from|that)\b',
        r'\bthe\s+(evidence|record|material)\s+(shows?|indicates?|reveals?)\b',
        r'\bupon\s+perusal\b',
        r'\bwe\s+are\s+in\s+agreement\b',
        r'\bwe\s+(do\s+not|cannot)\s+(agree|accept|find)\b',
        r'\bthe\s+(above|aforesaid|foregoing)\s+(discussion|analysis)\b',
        r'\bapplying\s+the\s+(law|test|principle)\b',
        r'\bthe\s+courts?\s+below\s+(has|have|had)\b',
        r'\bthe\s+(trial|high)\s+court\s+(has|have|correctly|rightly)\b',
    ]),
    ('FACT', [
        r'\bthe\s+facts?\s+(of\s+the\s+case|leading\s+to|briefly|in\s+(brief|nutshell))\b',
        r'\bthe\s+prosecution\s+(case|story|version)\b',
        r'\bbriefly\s+stated\b',
        r'\bthe\s+background\b',
        r'\bthe\s+appellant\s+(herein\s+)?(was|is|has\s+been)\b',
        r'\bthe\s+respondent\s+(herein\s+)?(was|is|has\s+been)\b',
        r'\bthe\s+impugned\s+(judgment|order)\b',
        r'\bthe\s+(first|second|trial|sessions|high)\s+court\s+(has|had|held)\b',
        r'\bvide\s+(judgment|order)\s+dated\b',
        r'\b(aggrieved|dissatisfied)\s+(by|with)\b',
        r'\bpassed\s+in\s+(criminal|civil)\s+appeal\b',
        r'\bregistered\s+under\s+section\b',
        r'\bthe\s+deceased\b',
        r'\bthe\s+accused\b',
        r'\bthe\s+(complaint|FIR|charge.?sheet)\b',
        r'\b(was|were)\s+(transferred|posted|appointed|promoted|suspended|dismissed)\b',
        r'\bby\s+(an?\s+)?(order|notification)\s+dated\b',
        r'\bon\s+\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b',
        r'\bthe\s+order\s+of\s+transfer\b',
        r'\bthe\s+(writ\s+petition|appeal|petition)\s+(was\s+)?(filed|moved|dismissed|allowed|disposed)\b',
        r'\bthe\s+(CAT|tribunal|high\s+court|sessions\s+court|magistrate)\b',
        r'\b(interim|ad\s+interim)\s+(stay|order|direction|relief)\b',
        r'\bthe\s+(police|inspector|sub.inspector)\b',
        r'\bpostmortem\b',
        r'\binjur(y|ies)\s+(found|noted|observed)\b',
        r'\bwitness(es)?\s+(deposed|stated|testified)\b',
        r'\bthe\s+(first|second|third|1st|2nd|3rd)\s+(accused|witness|complainant|informant)\b',
        r'\bleave\s+granted\b',
    ]),
]
_COMPILED_RULES = [(role, [re.compile(p, re.I | re.S) for p in patterns])
                    for role, patterns in _ROLE_RULES]
_HEADING_RE = re.compile(
    r'^(preliminary|background|facts?|evidence|submissions?|arguments?|'
    r'issues?|analysis|discussion|reasoning|conclusion|order|'
    r'the\s+relevant\s+facts?|the\s+evidence|the\s+submissions?)$', re.I)

# A paragraph in which the court rejects an argument ("We are not prepared
# to assume ... as was urged on behalf of the petitioner") is reasoning, not
# the party's argument, even though it matches ARG_P patterns. When one of
# these cues is present, ARG_P/ARG_R hits are counted as REASON instead.
_REJECTION_CUES = re.compile(
    r'\bwe\s+are\s+not\s+prepared\s+to\b|'
    r'\bwe\s+(do\s+not|does\s+not|cannot|are\s+unable\s+to)\s+(agree|accept|find)\b|'
    r'\bthis\s+(submission|argument|contention)\s+(ignores|overlooks|is\s+not\s+correct|is\s+laughable)\b|'
    r'\bwe\s+(reject|do\s+not\s+accept)\s+the\s+(submission|argument|contention)\b|'
    r'\bwe\s+find\s+no\s+(merit|substance)\s+in\b|'
    r'\bwe\s+(fail|are\s+unable)\s+to\s+see\b',
    re.I,
)

def _assign_role(para_text, para_number, total_paras):
    text = para_text.strip()
    if len(text.split()) <= 6 and _HEADING_RE.match(text):
        return 'NONE'
    scores = {role: 0 for role, _ in _COMPILED_RULES}
    for role, patterns in _COMPILED_RULES:
        for pat in patterns:
            if pat.search(text):
                scores[role] += 1

    # Court rejecting an argument -> REASON (see _REJECTION_CUES).
    if _REJECTION_CUES.search(text):
        for role in ('ARG_P', 'ARG_R'):
            if scores[role] > 0:
                scores['REASON'] += scores[role]
                scores[role] = 0

    # Position in the judgment is used only as a fallback when no pattern
    # matched, so it never overrides an actual content match.
    if total_paras > 0 and max(scores.values()) == 0:
        position = para_number / total_paras
        if position >= 0.90:
            scores['ORDER'] += 3; scores['HOLDING'] += 2
        elif position >= 0.75:
            scores['HOLDING'] += 1; scores['REASON'] += 1
        elif position >= 0.35:
            scores['REASON'] += 1
        elif position <= 0.30:
            scores['FACT'] += 2

    best_role = max(scores, key=scores.get)
    return best_role if scores[best_role] > 0 else 'NONE'

def _role_tag_paragraphs(paragraphs):
    total = len(paragraphs)
    for p in paragraphs:
        p['rhetorical_role'] = _assign_role(p['text'], p['number'], total)
    return paragraphs

def extract_pdf_segments(file_bytes):
    """Full pipeline for PDF: page extraction, header/footer stripping,
    LIS paragraph split, role tagging. No merging — one entry per real
    judgment paragraph, exactly as the LIS split identified it.
    Each returned segment also carries 'number' (the real judgment
    paragraph number, e.g. 40) and 'label' (same, but as the original
    "40" / "40.1" string)."""
    import pdfplumber
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        raw_pages = [p.extract_text(x_tolerance=3, y_tolerance=3) or "" for p in pdf.pages]
    n = len(raw_pages)
    pre = [re.sub(r'\n\d+\n', '\n', p) for p in raw_pages]

    running_header, running_footer, paged_pfx, has_pg_ftr, _pgnum = _detect_header_footer(pre, n)

    line_page_count = Counter()
    for page_text in pre:
        seen_on_page = set(l.strip() for l in page_text.split('\n') if l.strip() and len(l.strip()) > 15)
        for s in seen_on_page:
            line_page_count[s] += 1
    recur_threshold = max(3, n // 4)
    recurring_lines = {s for s, cnt in line_page_count.items() if cnt >= recur_threshold}

    all_lines = []
    for page_text in pre:
        for line in page_text.split('\n'):
            s = line.strip()
            if not s: continue
            if re.match(r'^\d+$', s): continue
            if running_header and s == running_header: continue
            if running_footer and s == running_footer: continue
            if paged_pfx and s.startswith(paged_pfx): continue
            if has_pg_ftr and _pgnum.match(s): continue
            if s in recurring_lines: continue
            all_lines.append(s)

    full_text = re.sub(r'\s+', ' ', ' '.join(all_lines)).strip()
    boundary = _find_para_boundary(full_text)
    body_text = full_text[boundary:] if boundary is not None else full_text

    cutoff = int(len(body_text) * 0.95)
    _sig = re.compile(r'\.{5,}\s*J\.|New\s+Delhi\b|\.\.\.\s*J\.', re.I)
    m_end = _sig.search(body_text[cutoff:])
    if m_end:
        body_text = body_text[:cutoff + m_end.start()].strip()

    paragraphs = _role_tag_paragraphs(_split_paragraphs_core(body_text))
    segments = [{'text': p['text'], 'role': p['rhetorical_role'],
                 'number': p['number'], 'label': p['label']} for p in paragraphs]
    return segments, n, len(paragraphs)

def extract_txt_segments(text):
    """Same paragraph-split + role-tag pipeline for plain text (no page
    structure, so no header/footer stage needed). No merging.
    Segments carry 'number'/'label' too — see extract_pdf_segments."""
    full_text = re.sub(r'\s+', ' ', text).strip()
    boundary = _find_para_boundary(full_text)
    body_text = full_text[boundary:] if boundary is not None else full_text
    paragraphs = _role_tag_paragraphs(_split_paragraphs_core(body_text))
    segments = [{'text': p['text'], 'role': p['rhetorical_role'],
                 'number': p['number'], 'label': p['label']} for p in paragraphs]
    return segments, len(paragraphs)

@st.cache_data(show_spinner=False)
def parse_uploaded_judgment_cached(file_bytes, file_name):
    """Returns (paragraphs, info, roles, numbers) based on file extension.
    `numbers` is a parallel list of the real judgment paragraph number for
    each parsed paragraph (TXT/PDF), or a 1-based positional fallback for
    JSON (which has no paragraph-number concept — field-path labels are
    used for splitting instead). `numbers` is what batch mode uses to
    match a CSV's `gold_para_num` to the right parsed paragraph.

    Cached on (file_bytes, file_name) so the LIS split + role-tagging
    pipeline runs ONCE per uploaded file, not on every Streamlit rerun
    (every slider move / checkbox toggle triggers a full script rerun —
    without this cache, the whole parse pipeline re-ran every time, which
    is why this was slow)."""
    name = file_name.lower()

    if name.endswith(".json"):
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        paragraphs = split_json_judgment(data)
        n_top = len(data) if isinstance(data, dict) else None
        info = f"JSON — {n_top} top-level fields -> {len(paragraphs)} paragraphs (recursive field-path split)."
        numbers = list(range(1, len(paragraphs) + 1))  # positional fallback; no real paragraph numbers in JSON
        return paragraphs, info, None, numbers

    elif name.endswith(".pdf"):
        segments, n_pages, n_raw_paras = extract_pdf_segments(file_bytes)
        paragraphs = [s["text"] for s in segments]
        roles = [s["role"] for s in segments]
        numbers = [s["number"] for s in segments]
        info = f"PDF — {n_pages} pages -> {len(paragraphs)} paragraphs (one per real judgment paragraph, no merging)."
        return paragraphs, info, roles, numbers

    else:  # .txt or unspecified
        text = file_bytes.decode("utf-8", errors="replace")
        segments, n_raw_paras = extract_txt_segments(text)
        paragraphs = [s["text"] for s in segments]
        roles = [s["role"] for s in segments]
        numbers = [s["number"] for s in segments]
        info = f"TXT — {len(paragraphs)} paragraphs (one per real judgment paragraph, no merging)."
        return paragraphs, info, roles, numbers

def parse_uploaded_judgment(uploaded_file):
    """Thin wrapper — extracts (bytes, name) so the cached function above
    has a stable, hashable cache key across Streamlit reruns."""
    return parse_uploaded_judgment_cached(uploaded_file.getvalue(), uploaded_file.name)

# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_bm25_index(doc_key, paragraphs_tuple):
    """Cached BM25 index for a document — reused across every query and
    every model/encoding combo for that document (BM25 doesn't depend on
    embedding model or encoding mode at all)."""
    return BM25Okapi([preprocess(p) for p in paragraphs_tuple])

@st.cache_resource(show_spinner=False)
def build_paragraph_embeddings(model_name, doc_key, paragraphs_tuple, encoding_mode_label, _version_key=0.0):
    """Cache key includes encoding_mode_label, so legacy and chunk-pool
    embeddings for the same model/document are cached separately and don't
    collide — this is what makes running both modes side by side cheap on
    a rerun (only the newly-selected combo re-encodes). `_version_key`
    (see _model_version_key) makes a retrained checkpoint invalidate the
    cached embeddings."""
    return encode_texts(model_name, paragraphs_tuple, encoding_mode_label, legacy_batch_size=32)

def compute_raw_normalized_scores(query, paragraphs, bm25, model_name, para_embeddings, encoding_mode_label):
    """Returns (bm25_norm, faiss_norm, bm25_raw, faiss_raw).

    The *_norm arrays are the existing min-max-stretched scores used for
    ranking (unchanged behavior). The *_raw arrays are the pre-normalization
    scores — these are what the relevance gate checks against, because
    min-max normalization by construction always stretches the best
    available match up to 1.0 regardless of whether it's actually a good
    match in absolute terms. Ranking logic itself is untouched; only the
    gate/warning layer looks at the raw values.

    The query is encoded via encode_texts() using the SAME encoding_mode_
    label as the paragraph embeddings it's being compared against — a
    Legacy-encoded query is never scored against Chunk-Pool-Mean-encoded
    paragraphs or vice versa.

    BM25 tokenization: uses preprocess(query) AS A LIST (duplicates
    preserved) — BM25Okapi's get_scores sums contribution per occurrence
    in the query token list, so this must match however batch mode tokenizes
    the same query for its own BM25-only ranking (see run_batch_mode), so
    that both modes give the same score for the same query.
    """
    n = len(paragraphs)
    q_toks = preprocess(query)
    bm25_raw = np.array(bm25.get_scores(q_toks)) if q_toks else np.zeros(n)
    if query.strip():
        q_emb = encode_texts(model_name, [query], encoding_mode_label)[0]
        faiss_raw = para_embeddings @ q_emb
    else:
        faiss_raw = np.zeros(n)
    return _minmax(bm25_raw), _minmax(faiss_raw), bm25_raw, faiss_raw

def combine_and_rank(bm25_norm, faiss_norm, bm25_weight):
    combined = bm25_weight * bm25_norm + (1.0 - bm25_weight) * faiss_norm
    order = np.argsort(combined)[::-1]
    return order, combined

def passes_relevance_gate(bm25_score, cosine_score, term_overlap, cosine_min, bm25_min, min_term_overlap):
    """OR gate on RAW (pre-minmax) scores for the SPECIFIC paragraph being
    displayed (top_idx) — mirrors Stage 1's alignment acceptance gate
    (cosine OR bm25, not AND), but evaluated per-paragraph, not per-document.

    Checks the displayed paragraph's own raw scores, not a document-wide
    max — a strong raw score on some unrelated paragraph elsewhere in the
    document must not vouch for a different, weak paragraph the combined
    ranking happened to display.

    The BM25 branch also requires a minimum number of DISTINCT query terms
    present in the paragraph (term_overlap), not just a raw score above a
    floor — guards against BM25 systematically over-scoring short,
    content-free paragraphs on a single coincidental matched word.

    cosine_min is the per-(model, encoding-mode) floor set in the UI —
    raw score scales differ across embedding spaces AND potentially across
    encoding modes (see module docstring's score-scale caveat). bm25_min
    and min_term_overlap are shared, since BM25 runs on the same token
    index regardless of embedding model or encoding mode.

    A raw score of exactly 0 never counts as clearing a floor, even when
    the floor itself is set to 0.
    """
    bm25_ok = (bm25_score > 0.0 and bm25_score >= bm25_min
               and term_overlap >= min_term_overlap)
    cosine_ok = cosine_score > 0.0 and cosine_score >= cosine_min
    return cosine_ok or bm25_ok

def score_gap(combined):
    """Gap between the top-1 and top-2 combined (post-minmax) scores. A
    small gap means the ranking is nearly flat at the top — informational
    only, since a genuine near-tie between two good paragraphs looks the
    same as a flat 'nothing really stands out' distribution and can't be
    told apart from score alone; surfaced as a caption, not a warning."""
    if len(combined) < 2:
        return None
    ordered = np.sort(combined)[::-1]
    return float(ordered[0] - ordered[1])

def _parse_extra_paras(val):
    """Parses `additional_relevant_paras` cell (comma/semicolon-separated
    paragraph numbers, e.g. "104; 107") into a list of ints. Blank/NaN ->
    empty list."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    out = []
    for part in re.split(r'[;,]\s*', s):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            pass
    return out

# winner_(lower_rank) and verdict_reason are computed together here so the
# two columns of the exported CSV always agree.
def _score_and_explain(rank_faiss, rank_bm25, n_matched_gold, gold_found):
    """Returns (winner, reason) for one query/model/encoding row, computed
    together from the same underlying facts so the two fields shown in the
    exported CSV can never contradict each other."""
    if not gold_found:
        return None, "Gold paragraph not found in this document's parse — not comparable."
    if rank_faiss is None and rank_bm25 is None:
        return None, "Rank unavailable for both retrievers — not comparable."
    if rank_faiss is None:
        return "bm25 (faiss missed)", f"BM25 found it at rank {rank_bm25}; FAISS did not rank it in range."
    if rank_bm25 is None:
        return "faiss (bm25 missed)", f"FAISS found it at rank {rank_faiss}; BM25 did not rank it in range."
    if rank_bm25 < rank_faiss:
        if n_matched_gold >= 2:
            return "bm25", (f"BM25 wins (rank {rank_bm25} vs {rank_faiss}) — {n_matched_gold} shared "
                             f"keyword(s) gave it a strong, direct lexical anchor.")
        return "bm25", (f"BM25 wins (rank {rank_bm25} vs {rank_faiss}) despite only {n_matched_gold} "
                         f"shared keyword(s) — a narrow lexical hit still outranked semantic similarity here.")
    if rank_faiss < rank_bm25:
        if n_matched_gold == 0:
            return "faiss", (f"FAISS wins (rank {rank_faiss} vs {rank_bm25}) — zero lexical overlap with "
                              f"the gold paragraph; only semantic similarity could have found it.")
        return "faiss", (f"FAISS wins (rank {rank_faiss} vs {rank_bm25}) — semantic similarity outranked "
                          f"BM25's {n_matched_gold} shared keyword(s), likely diluted by more frequent "
                          f"matches for those same words elsewhere in the document.")
    return "tie", f"Tie — both retrievers ranked the gold paragraph at #{rank_faiss}."

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE JUDGMENT, INTERACTIVE QUERY (original tool, unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

def run_single_query_mode():
    st.markdown("### Step 1 — Upload a judgment")
    uploaded = st.file_uploader("PDF, TXT, or structured JSON", type=["pdf", "txt", "json"], key="single_upload")
    if uploaded is None:
        st.info("Upload a file to continue. Supported: .pdf, .txt, .json"); return

    try:
        paragraphs, info, roles, numbers = parse_uploaded_judgment(uploaded)
    except Exception as e:
        st.error(f"Failed to parse '{uploaded.name}': {e}")
        st.caption("If this is a PDF: `pip install pdfplumber` may be needed. If JSON: confirm it's valid JSON.")
        return

    st.success(info)
    if len(paragraphs) < 3:
        st.warning(
            f"Only {len(paragraphs)} paragraph(s) parsed — the splitter may not have found "
            f"the structure it expected. Check Step 2 below; if it looks wrong, the document "
            f"may not use numbered paragraphs ('1.', '2.', ...) or may need a different format."
        )

    st.markdown("### Step 2 — Parsed paragraphs (sanity check before querying)")
    with st.expander(f"{len(paragraphs)} paragraphs — click to inspect", expanded=len(paragraphs) < 3):
        for i, p in enumerate(paragraphs, 1):
            role_tag = f" `{roles[i-1]}`" if roles else ""
            num_tag = f" ¶{numbers[i-1]}" if numbers and numbers[i-1] not in (0, None) else ""
            st.markdown(f"**[{i}]**{num_tag}{role_tag} {p[:400]}{'…' if len(p) > 400 else ''}")

    st.markdown("### Step 3 — Model(s), encoding mode(s), & weight")
    st.caption(
        "Pick one or more base models AND one or more encoding modes — every combination "
        "is run and shown side by side. Chunk-Pool Mean chunks any paragraph exceeding the "
        "model's token budget and mean-pools the chunk embeddings, same as training/eval; "
        "Legacy truncates at the token budget like before. Defaults below compare both "
        "encoding modes on both fine-tuned models (2×2), which directly answers "
        "'does the training method matter' vs. 'does the encoding method matter' for this "
        "specific document."
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        _default_models = [k for k in EMBED_MODELS if k.startswith("Mine_Finetuned")]
        selected_models = st.multiselect(
            "Base embedding model(s) to compare", list(EMBED_MODELS.keys()),
            default=_default_models or [list(EMBED_MODELS.keys())[-1]], key="single_models",
        )
        selected_modes = st.multiselect(
            "Encoding mode(s) at inference", list(ENCODING_MODES.keys()),
            default=list(ENCODING_MODES.keys()), key="single_modes",
        )
    with col2:
        bm25_weight = st.slider("BM25 weight (FAISS weight = 1 − this)", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"BM25={bm25_weight:.2f} · FAISS={1-bm25_weight:.2f}")

    if not selected_models or not selected_modes:
        st.warning("Select at least one model and one encoding mode to continue.")
        return

    # (combined_label, model_path, encoding_mode_label, base_label)
    models_to_run = [
        (f"{m_label} — {enc_label}", EMBED_MODELS[m_label], enc_label, m_label)
        for m_label in selected_models
        for enc_label in selected_modes
    ]

    with st.expander("⚙️ Relevance-gate settings (advanced — calibrate against your own labeled queries)"):
        st.caption(
            "A paragraph is only shown as a *confirmed* match if its RAW (pre-normalization) "
            "cosine similarity or BM25 score clears a floor — the same OR rule as the Stage 1 alignment "
            "gate (cosine OR bm25, not AND). **Cosine floors are set per (model, encoding "
            "mode) combination**, not just per model: raw score scales differ across "
            "embedding spaces (a model without SBERT-style contrastive tuning tends to occupy "
            "a narrower cone of the embedding space, inflating cosine similarity for unrelated "
            "text) AND potentially across encoding modes (Chunk-Pool Mean's raw dot product "
            "is not guaranteed to be on the same normalized scale as Legacy's — see the module "
            "docstring). Defaults below are starting points, not validated numbers — tune them "
            "against a labeled set of relevant / irrelevant query-document pairs, the same way "
            "bm25_min_score and role_bonus were calibrated for Stage 1. BM25's floor stays "
            "shared, since BM25 runs on the same token index regardless of model or encoding mode."
        )
        st.markdown("**Per-(model, encoding) raw-cosine floor:**")
        cosine_min_by_combo = {}
        cc_cols = st.columns(max(len(models_to_run), 1))
        for i, (combined_label, _, _, base_label) in enumerate(models_to_run):
            default_c = DEFAULT_COSINE_MIN_BY_MODEL.get(base_label, DEFAULT_COSINE_MIN_FALLBACK)
            with cc_cols[i % len(cc_cols)]:
                cosine_min_by_combo[combined_label] = st.slider(
                    combined_label, 0.0, 1.0, default_c, 0.01, key=f"cosine_min_{combined_label}"
                )
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            bm25_min = st.slider("Min raw BM25 score (shared across models)", 0.0, 20.0, DEFAULT_BM25_MIN_RAW_THRESHOLD, 0.5)
        with gc2:
            gap_min = st.slider("Min score-gap (rank1 vs rank2) to call it a clear leader", 0.0, 0.5, DEFAULT_SCORE_GAP_MIN, 0.01)
        with gc3:
            min_term_overlap = st.slider(
                "Min distinct query terms matched (BM25 branch)", 0, 5, DEFAULT_MIN_TERM_OVERLAP, 1,
                help="Guards against a short paragraph clearing the BM25 score floor on a single "
                     "coincidental word (e.g. a generic sentence matching only 'issue' out of "
                     "'issue estoppel') — requires this many distinct query terms to actually appear."
            )

    st.markdown("### Step 4 — Query")
    query = st.text_input("Enter your search query", placeholder="e.g. grounds for issue estoppel in non-criminal proceedings")
    show_grid = st.checkbox("Also show top-1 paragraph's rank across the full 5-point weight grid (robustness check)", value=False)
    run = st.button("Search", type="primary", disabled=not query.strip())

    # Streamlit reruns the whole script on every interaction, and st.button()
    # is True only on the rerun its own click triggers. Keep the submitted
    # query in session_state so results survive later interactions (e.g. the
    # download button) and update live when a slider is moved.
    if run:
        st.session_state["single_committed_query"] = query

    committed_query = st.session_state.get("single_committed_query")
    if not committed_query:
        return

    doc_key = f"{uploaded.name}:{len(paragraphs)}"
    bm25 = build_bm25_index(doc_key, tuple(paragraphs))

    # ── Unmissable banner: which weight these results are computed at ──
    st.markdown("---")
    st.warning(
        f"### 📍 Results below are computed at: **BM25 weight = {bm25_weight:.2f}, "
        f"FAISS weight = {1 - bm25_weight:.2f}**\n"
        f"Change the slider in Step 3 to update the results below live, or type a new "
        f"query and click Search again."
    )

    all_download_rows = []

    for combined_label, model_name, encoding_mode_label, base_label in models_to_run:
        st.markdown("---")
        st.subheader(f"{combined_label}  ·  weight (BM25={bm25_weight:.2f}, FAISS={1-bm25_weight:.2f})")
        with st.spinner(f"Encoding {len(paragraphs)} paragraphs with {combined_label}…"):
            para_emb = build_paragraph_embeddings(
                model_name, doc_key, tuple(paragraphs), encoding_mode_label,
                _model_version_key(model_name),
            )
        bm25_norm, faiss_norm, bm25_raw, faiss_raw = compute_raw_normalized_scores(
            committed_query, paragraphs, bm25, model_name, para_emb, encoding_mode_label
        )
        order, combined = combine_and_rank(bm25_norm, faiss_norm, bm25_weight)

        top_idx = order[0]
        role_tag = f" · role: `{roles[top_idx]}`" if roles else ""

        # ── Relevance gate: honest labeling instead of unearned confidence ──
        # Evaluated on the SPECIFIC displayed paragraph (top_idx)'s own raw
        # scores — not a document-wide max, which could belong to a
        # different, unshown paragraph once both weights are nonzero.
        combo_cosine_min = cosine_min_by_combo.get(combined_label, DEFAULT_COSINE_MIN_FALLBACK)
        top_term_overlap = len(set(preprocess(committed_query)) & set(preprocess(paragraphs[top_idx])))
        gate_ok = passes_relevance_gate(
            float(bm25_raw[top_idx]), float(faiss_raw[top_idx]), top_term_overlap,
            combo_cosine_min, bm25_min, min_term_overlap
        )
        gap = score_gap(combined)

        if not gate_ok:
            st.error(
                "⚠️ **No paragraph clearly matches this query.** Neither the best raw cosine "
                "similarity nor the best raw BM25 score clears its floor — this document may "
                "simply not address what was asked. The closest match is shown below for "
                "reference only; treat it as unconfirmed, not as an answer."
            )
        elif gap is not None and gap < gap_min:
            st.caption(
                f"ℹ️ Top-1 and top-2 scores are close (gap={gap:.3f}) — no single paragraph "
                f"stands out clearly; worth checking the next candidates below too."
            )

        st.markdown(f"**Most relevant paragraph (rank 1 of {len(paragraphs)}) "
                     f"at BM25={bm25_weight:.2f}/FAISS={1-bm25_weight:.2f}{role_tag}:**")
        st.info(paragraphs[top_idx])
        st.caption(
            f"combined={combined[top_idx]:.4f} · bm25_norm={bm25_norm[top_idx]:.4f} · "
            f"faiss_norm={faiss_norm[top_idx]:.4f}  ‖  "
            f"raw: bm25={bm25_raw[top_idx]:.3f} (floor {bm25_min:.1f}, needs ≥{min_term_overlap} matched terms — "
            f"this paragraph matched {top_term_overlap}) · "
            f"cosine={faiss_raw[top_idx]:.3f} (floor {combo_cosine_min:.2f}, this model+encoding) · "
            f"gate={'✅ passed' if gate_ok else '❌ not met'}"
        )

        with st.expander(f"Next {min(9, len(order)-1)} candidates"):
            for rank, idx in enumerate(order[1:10], start=2):
                st.markdown(f"**[{rank}]** (combined={combined[idx]:.4f}) {paragraphs[idx][:300]}"
                            f"{'…' if len(paragraphs[idx]) > 300 else ''}")

        for rank, idx in enumerate(order[:10], start=1):
            all_download_rows.append({
                "Query": committed_query,
                "Model": base_label, "Encoding": encoding_mode_label,
                "BM25 weight": bm25_weight, "FAISS weight": round(1 - bm25_weight, 2),
                "Rank": rank, "Paragraph #": idx + 1,
                "Judgment paragraph number": numbers[idx] if numbers else "",
                "Role": roles[idx] if roles else "",
                "Combined score": round(float(combined[idx]), 4),
                "BM25 norm": round(float(bm25_norm[idx]), 4), "FAISS norm": round(float(faiss_norm[idx]), 4),
                "BM25 raw": round(float(bm25_raw[idx]), 4), "FAISS raw (cosine)": round(float(faiss_raw[idx]), 4),
                "Relevance gate passed (top-1 only)": gate_ok if rank == 1 else "",
                "Top1/Top2 score gap": round(gap, 4) if (rank == 1 and gap is not None) else "",
                # Logged so a run can be audited later without guessing what the
                # sliders were set to.
                "Cosine floor used (this model+encoding)": combo_cosine_min,
                "BM25 floor used (shared)": bm25_min,
                "Gap floor used": gap_min,
                "Query terms matched in paragraph (top-1 only)": top_term_overlap if rank == 1 else "",
                "Min term overlap required": min_term_overlap,
                "Paragraph text": paragraphs[idx],
            })

        if show_grid:
            st.markdown("**Rank of this model's top-1 paragraph across the weight grid "
                         "(for comparison only — results above stay at the slider's weight):**")
            grid_rows = []
            for bw, fw in WEIGHT_GRID:
                g_order, _ = combine_and_rank(bm25_norm, faiss_norm, bw)
                rank_at_w = list(g_order).index(top_idx) + 1
                marker = " 👈 current" if abs(bw - bm25_weight) < 1e-9 else ""
                grid_rows.append(f"({bw},{fw})\u2192rank {rank_at_w}{marker}")
            st.caption("  ·  ".join(grid_rows))

    st.markdown("---")
    st.markdown(f"### 💾 Save these results (BM25={bm25_weight:.2f}, FAISS={1-bm25_weight:.2f})")
    download_df = pd.DataFrame(all_download_rows)
    query_slug = re.sub(r"[^a-zA-Z0-9]+", "_", committed_query.strip())[:40].strip("_")
    st.download_button(
        f"Download top-10-per-combo CSV — query='{committed_query[:40]}', weight=({bm25_weight:.2f},{1-bm25_weight:.2f})",
        download_df.to_csv(index=False),
        f"evidence_extraction_{query_slug}_bm25_{bm25_weight:.2f}_{uploaded.name}.csv",
        "text/csv",
        # Fixed key: the label changes with the query, which would otherwise
        # change the widget's auto-generated id between reruns.
        key="single_query_download_btn",
    )

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — BATCH CSV EVALUATION: FAISS-only (0,1) vs BM25-only (1,0)
# ══════════════════════════════════════════════════════════════════════════════

def run_batch_mode():
    st.header("Batch CSV evaluation")
    st.caption(
        "Upload a query test-set CSV (one row per query, each pointing at a paragraph number "
        "in a specific judgment) plus the judgment file(s) it refers to. For every query this "
        "computes where the gold paragraph lands in the ranking under **FAISS-only** "
        "(BM25 weight=0.0, FAISS weight=1.0) and **BM25-only** (BM25 weight=1.0, FAISS "
        "weight=0.0) — the two pure-retriever endpoints of the weight grid — for every "
        "selected model × encoding-mode combination, then reports Accuracy@k / MRR aggregates. "
        "Every aggregate table also reports **coverage** — how many queries' gold paragraphs "
        "were actually locatable in the parse — since accuracy is computed conditional on that, "
        "not silently over a smaller denominator with no visible signal."
    )

    st.markdown("#### Step 1 — Upload the query CSV")
    st.caption(
        "Expected columns: `query_id`, `judgment_id`, `query_text` (aliases `query`/`Query` "
        "accepted), `gold_para_num` (aliases `gold_paragraph_number`/`para_num`/`paragraph`/"
        "`Paragraph #`/`gold_para`/`gold_paragraph` accepted). Optional: `additional_relevant_paras` "
        "(comma/semicolon-separated extra paragraph numbers that also count as a hit), "
        "`query_length_class` (alias `query_length`), `query_style` (alias `query_type`) — used "
        "for breakdown reporting if present."
    )
    query_file = st.file_uploader("Query CSV", type=["csv"], key="batch_query_csv")

    st.markdown("#### Step 2 — Upload the judgment file(s) referenced by the CSV")
    st.caption(
        "PDF or TXT, one file per judgment. The tool guesses which `judgment_id` each file "
        "belongs to from its filename (e.g. `C14.txt` → `C14`) — confirm or fix the mapping "
        "below before running."
    )
    judgment_files = st.file_uploader(
        "Judgment file(s)", type=["pdf", "txt"], accept_multiple_files=True, key="batch_judgment_files"
    )

    if not query_file or not judgment_files:
        st.info("Upload both a query CSV and at least one judgment file to continue.")
        return

    try:
        raw_df = pd.read_csv(query_file)
    except Exception as e:
        st.error(f"Could not read the query CSV: {e}")
        return

    colmap = {}
    for target, aliases in [
        ("query_id", ["query_id", "id", "Query_ID"]),
        ("judgment_id", ["judgment_id", "judgment", "doc_id", "Judgment_ID"]),
        ("query_text", ["query_text", "query", "Query"]),
        ("gold_para_num", ["gold_para_num", "gold_paragraph_number", "para_num", "paragraph",
                            "Paragraph #", "gold_para", "gold_paragraph"]),
    ]:
        for a in aliases:
            if a in raw_df.columns:
                colmap[target] = a
                break

    missing = [t for t in ["judgment_id", "query_text", "gold_para_num"] if t not in colmap]
    if missing:
        st.error(f"Query CSV is missing required column(s) (or a recognized alias): {missing}. "
                  f"Columns found: {list(raw_df.columns)}")
        return

    df = pd.DataFrame({
        "query_id": raw_df[colmap["query_id"]] if "query_id" in colmap else [f"q{i+1}" for i in range(len(raw_df))],
        "judgment_id": raw_df[colmap["judgment_id"]].astype(str),
        "query_text": raw_df[colmap["query_text"]].astype(str),
        "gold_para_num": raw_df[colmap["gold_para_num"]],
        "query_length_class": raw_df["query_length_class"] if "query_length_class" in raw_df.columns else (
            raw_df["query_length"] if "query_length" in raw_df.columns else ""),
        "query_style": raw_df["query_style"] if "query_style" in raw_df.columns else (
            raw_df["query_type"] if "query_type" in raw_df.columns else ""),
        "additional_relevant_paras": raw_df["additional_relevant_paras"] if "additional_relevant_paras" in raw_df.columns else "",
    })
    st.success(f"Loaded {len(df)} queries across judgment_id(s): {sorted(df['judgment_id'].unique())}")

    st.markdown("#### Step 3 — Confirm file → judgment_id mapping")
    unique_jids = sorted(df["judgment_id"].unique())
    file_to_jid = {}
    map_cols = st.columns(min(3, len(judgment_files)) or 1)
    for i, f in enumerate(judgment_files):
        stem = os.path.splitext(f.name)[0]
        guess = stem if stem in unique_jids else next(
            (j for j in unique_jids if j.lower() == stem.lower()), unique_jids[0] if unique_jids else stem
        )
        options = unique_jids + ["— skip this file —"]
        default_idx = options.index(guess) if guess in options else 0
        with map_cols[i % len(map_cols)]:
            choice = st.selectbox(f"`{f.name}` →", options, index=default_idx, key=f"map_{f.name}")
        if choice != "— skip this file —":
            file_to_jid[choice] = f

    unmapped_jids = [j for j in unique_jids if j not in file_to_jid]
    if unmapped_jids:
        st.warning(f"No judgment file mapped for judgment_id(s) {unmapped_jids} — "
                   f"their queries will be skipped.")

    st.markdown("#### Step 4 — Model(s) & encoding mode(s)")
    _default_models = [k for k in EMBED_MODELS if k.startswith("Mine_Finetuned")]
    selected_models = st.multiselect(
        "Base embedding model(s)", list(EMBED_MODELS.keys()),
        default=_default_models or [list(EMBED_MODELS.keys())[-1]], key="batch_models",
    )
    selected_modes = st.multiselect(
        "Encoding mode(s)", list(ENCODING_MODES.keys()), default=list(ENCODING_MODES.keys()),
        key="batch_modes",
    )
    if not selected_models or not selected_modes:
        st.warning("Select at least one model and one encoding mode.")
        return

    models_to_run = [
        (f"{m_label} — {enc_label}", EMBED_MODELS[m_label], enc_label, m_label)
        for m_label in selected_models for enc_label in selected_modes
    ]

    run = st.button("Run batch evaluation", type="primary")

    # Only the expensive evaluation runs on the button click; its results are
    # kept in session_state so the tables and download buttons below survive
    # later reruns.
    if run:
        st.session_state["batch_results_df"] = _run_batch_evaluation(file_to_jid, eval_target_df=df, models_to_run=models_to_run)

    results_df = st.session_state.get("batch_results_df")
    if results_df is None or results_df.empty:
        return

    _render_batch_results(results_df)


def _run_batch_evaluation(file_to_jid, eval_target_df, models_to_run):
    """Runs the expensive part of batch evaluation: parses every mapped
    judgment, builds/embeds every (judgment, model, encoding) combination,
    and scores every query. Returns the raw per-query results_df, which the
    caller caches in st.session_state.
    """
    df = eval_target_df

    # ── Parse every mapped judgment once ──
    judgment_data = {}
    with st.spinner("Parsing judgment file(s)…"):
        for jid, f in file_to_jid.items():
            try:
                paragraphs, info, roles, numbers = parse_uploaded_judgment(f)
            except Exception as e:
                st.error(f"Failed to parse '{f.name}' (judgment_id={jid}): {e}")
                continue
            number_to_idx = {}
            for idx, num in enumerate(numbers):
                if num in (0, None):
                    continue
                number_to_idx.setdefault(int(num), idx)
            doc_key = f"{f.name}:{len(paragraphs)}"
            bm25_index = build_bm25_index(doc_key, tuple(paragraphs))
            judgment_data[jid] = {
                "paragraphs": paragraphs, "roles": roles, "numbers": numbers,
                "number_to_idx": number_to_idx, "doc_key": doc_key, "bm25": bm25_index,
                "info": info, "filename": f.name,
            }
            st.caption(f"`{f.name}` → judgment_id `{jid}`: {info}")

    eval_df = df[df["judgment_id"].isin(judgment_data.keys())].reset_index(drop=True)
    n_skipped = len(df) - len(eval_df)
    if n_skipped:
        st.caption(f"Skipping {n_skipped} quer(y/ies) whose judgment_id has no mapped file.")

    if eval_df.empty:
        st.error("No queries to evaluate — check the file → judgment_id mapping above.")
        return pd.DataFrame()

    # ── Pre-compute paragraph embeddings for every (judgment, model, encoding) combo ──
    embeddings_cache = {}
    with st.spinner("Encoding paragraphs for every judgment × model × encoding combination…"):
        for jid, jdata in judgment_data.items():
            for combined_label, model_name, enc_label, base_label in models_to_run:
                key = (jid, combined_label)
                embeddings_cache[key] = build_paragraph_embeddings(
                    model_name, jdata["doc_key"], tuple(jdata["paragraphs"]), enc_label,
                    _model_version_key(model_name),
                )

    # ── Run every query through every model/encoding combo, twice: FAISS-only + BM25-only ──
    results = []
    progress = st.progress(0.0)
    total = max(len(eval_df) * len(models_to_run), 1)
    step = 0
    for _, row in eval_df.iterrows():
        jid = row["judgment_id"]
        jdata = judgment_data[jid]
        paragraphs = jdata["paragraphs"]
        numbers = jdata["numbers"]
        number_to_idx = jdata["number_to_idx"]
        bm25_index = jdata["bm25"]

        try:
            gold_num = int(float(row["gold_para_num"]))
        except (ValueError, TypeError):
            gold_num = None
        extra_nums = _parse_extra_paras(row.get("additional_relevant_paras", ""))
        acceptable_nums = [n for n in ([gold_num] + extra_nums) if n is not None]
        acceptable_idxs = {number_to_idx[n] for n in acceptable_nums if n in number_to_idx}
        gold_found = len(acceptable_idxs) > 0
        if not gold_found:
            st.caption(f"⚠️ {row['query_id']}: gold paragraph #{gold_num} not found in parsed "
                       f"`{jdata['filename']}` (parser may have split it differently) — skipping rank.")

        # ── BM25-only ranking + matched-keyword info: computed ONCE per query,
        # not per model/encoding combo, since BM25 never touches an embedding
        # model. Query tokens are kept as a list (duplicates preserved), as
        # in compute_raw_normalized_scores, so both paths give the same BM25
        # score; the distinct-term set is used only to report matched terms. ──
        q_tokens_list = preprocess(row["query_text"])
        q_terms_set = set(q_tokens_list)
        bm25_raw_scores = np.array(bm25_index.get_scores(q_tokens_list)) if q_tokens_list else np.zeros(len(paragraphs))
        bm25_only_order = np.argsort(bm25_raw_scores)[::-1]

        gold_idx = number_to_idx.get(gold_num) if gold_num is not None else None
        matched_terms_at_gold = sorted(q_terms_set & set(preprocess(paragraphs[gold_idx]))) if gold_idx is not None else []

        bm25_top1_idx = int(bm25_only_order[0]) if len(bm25_only_order) else None
        bm25_top1_para_num = numbers[bm25_top1_idx] if bm25_top1_idx is not None else None
        matched_terms_at_bm25_top1 = (
            sorted(q_terms_set & set(preprocess(paragraphs[bm25_top1_idx]))) if bm25_top1_idx is not None else []
        )

        def _best_rank_bm25():
            if not gold_found:
                return None
            pos = [i for i, ix in enumerate(bm25_only_order) if ix in acceptable_idxs]
            return (min(pos) + 1) if pos else None

        rank_bm25 = _best_rank_bm25()

        for combined_label, model_name, enc_label, base_label in models_to_run:
            step += 1
            progress.progress(min(step / total, 1.0))
            para_emb = embeddings_cache[(jid, combined_label)]
            bm25_norm, faiss_norm, bm25_raw, faiss_raw = compute_raw_normalized_scores(
                row["query_text"], paragraphs, bm25_index, model_name, para_emb, enc_label
            )
            order_faiss, _ = combine_and_rank(bm25_norm, faiss_norm, 0.0)   # FAISS-only (0,1)

            def _best_rank_faiss():
                if not gold_found:
                    return None
                pos = [i for i, ix in enumerate(order_faiss) if ix in acceptable_idxs]
                return (min(pos) + 1) if pos else None

            rank_faiss = _best_rank_faiss()

            # NaN when the gold paragraph was not found in the parse; such
            # rows are reported through the coverage columns below.
            def _hit(rank, k):
                if rank is None:
                    return np.nan
                return rank <= k

            def _rr(rank):
                return (1.0 / rank) if rank else 0.0

            winner, verdict_reason = _score_and_explain(rank_faiss, rank_bm25, len(matched_terms_at_gold), gold_found)

            results.append({
                "query_id": row["query_id"], "judgment_id": jid,
                "query_style": row.get("query_style", ""), "query_length_class": row.get("query_length_class", ""),
                "model": base_label, "encoding": enc_label,
                "query_text": row["query_text"], "gold_para_num": gold_num,
                "gold_paragraph_found_in_parse": gold_found,
                "rank_faiss_only": rank_faiss, "rank_bm25_only": rank_bm25,
                "hit@1_faiss": _hit(rank_faiss, 1), "hit@3_faiss": _hit(rank_faiss, 3),
                "hit@5_faiss": _hit(rank_faiss, 5), "hit@10_faiss": _hit(rank_faiss, 10),
                "hit@1_bm25": _hit(rank_bm25, 1), "hit@3_bm25": _hit(rank_bm25, 3),
                "hit@5_bm25": _hit(rank_bm25, 5), "hit@10_bm25": _hit(rank_bm25, 10),
                "rr_faiss": _rr(rank_faiss), "rr_bm25": _rr(rank_bm25),
                "winner_(lower_rank)": winner,
                # ── BM25 keyword-matching transparency (model-independent) ──
                "bm25_query_terms_after_preprocessing": ", ".join(sorted(q_terms_set)),
                "bm25_matched_terms_at_gold_para": ", ".join(matched_terms_at_gold),
                "bm25_n_matched_terms_at_gold": len(matched_terms_at_gold),
                "bm25_actual_top1_para_num": bm25_top1_para_num,
                "bm25_matched_terms_at_its_top1": ", ".join(matched_terms_at_bm25_top1),
                "bm25_n_matched_terms_at_top1": len(matched_terms_at_bm25_top1),
                "verdict_reason": verdict_reason,
            })

    return pd.DataFrame(results)


def _render_batch_results(results_df):
    """Renders the aggregation/tables/download-buttons section of batch
    mode from a cached results_df, so reruns do not repeat the encoding."""

    # ══════════════════════════════════════════════════════════════
    # OVERALL VERDICT — presentable "which one wins and why" summary.
    # Compares BM25 against the best semantic rank any selected model
    # achieved per query (an upper bound for semantic retrieval), broken
    # down by lexical overlap. Table 1 of the paper instead compares BM25
    # with a single model; see evaluation/summarize_controlled_eval.py. ──
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🏁 Overall verdict — FAISS vs. BM25")

    comparable = results_df[results_df["gold_paragraph_found_in_parse"]]
    n_total_queries = results_df["query_id"].nunique()
    n_comparable_queries = comparable["query_id"].nunique()
    st.caption(
        f"Coverage: gold paragraph located in the parse for **{n_comparable_queries} / "
        f"{n_total_queries}** unique queries ({n_comparable_queries / max(n_total_queries, 1):.0%}). "
        f"Everything below is computed over the locatable subset only."
    )
    if comparable.empty:
        st.warning("No queries had a locatable gold paragraph — no verdict to compute.")
    else:
        best_faiss = comparable.groupby("query_id")["rank_faiss_only"].min()
        bm25_per_q = comparable.drop_duplicates("query_id").set_index("query_id")["rank_bm25_only"]
        overlap_per_q = comparable.drop_duplicates("query_id").set_index("query_id")["bm25_n_matched_terms_at_gold"]
        verdict_df = pd.DataFrame({
            "best_rank_faiss": best_faiss, "rank_bm25": bm25_per_q, "n_matched_terms": overlap_per_q,
        }).dropna()
        verdict_df["winner"] = np.where(
            verdict_df["best_rank_faiss"] < verdict_df["rank_bm25"], "faiss",
            np.where(verdict_df["rank_bm25"] < verdict_df["best_rank_faiss"], "bm25", "tie")
        )

        n = len(verdict_df)
        n_faiss = int((verdict_df["winner"] == "faiss").sum())
        n_bm25 = int((verdict_df["winner"] == "bm25").sum())
        n_tie = int((verdict_df["winner"] == "tie").sum())

        overall_winner = "FAISS (semantic)" if n_faiss > n_bm25 else ("BM25 (lexical)" if n_bm25 > n_faiss else "Tied")
        st.success(
            f"### 🏆 Overall winner across {n} queries: **{overall_winner}**\n"
            f"FAISS (best available model) wins **{n_faiss}/{n}** ({n_faiss/n:.0%}) · "
            f"BM25 wins **{n_bm25}/{n}** ({n_bm25/n:.0%}) · tie **{n_tie}/{n}** ({n_tie/n:.0%})"
        )

        st.markdown("**Why — win rate by lexical overlap with the gold paragraph:**")
        st.caption(
            "This is the mechanism, not just the outcome: BM25's score is built directly from "
            "shared keywords, so its win rate should track overlap closely. If FAISS is still "
            "winning even at high overlap, that's semantic understanding adding value beyond "
            "keyword matching, not just compensating for its absence."
        )

        def _bucket(n_matched):
            if n_matched == 0: return "0 (zero overlap)"
            if n_matched <= 2: return "1–2 (low overlap)"
            return "3+ (high overlap)"

        verdict_df["overlap_bucket"] = verdict_df["n_matched_terms"].apply(_bucket)
        bucket_order = ["0 (zero overlap)", "1–2 (low overlap)", "3+ (high overlap)"]
        bucket_summary = verdict_df.groupby("overlap_bucket").agg(
            n_queries=("winner", "count"),
            faiss_win_rate=("winner", lambda s: (s == "faiss").mean()),
            bm25_win_rate=("winner", lambda s: (s == "bm25").mean()),
            tie_rate=("winner", lambda s: (s == "tie").mean()),
        ).round(3).reindex(bucket_order).dropna(how="all").reset_index()
        st.dataframe(bucket_summary, use_container_width=True)

        st.download_button("Download overall verdict CSV", verdict_df.reset_index().rename(
            columns={"index": "query_id"}).to_csv(index=False),
            "batch_eval_overall_verdict.csv", "text/csv", key="batch_verdict_download_btn")

    st.markdown("---")
    st.markdown("### Per-query results")
    st.caption(
        "`verdict_reason` — a plain-language explanation of why FAISS or BM25 won *this specific "
        "model/encoding row* (rank comparison + lexical overlap at the gold paragraph), computed "
        "together with `winner_(lower_rank)` so the two columns can never contradict each other. "
        "`bm25_matched_terms_at_gold_para` — which preprocessed query terms (lowercased, "
        "stopwords removed, non-alphabetic tokens dropped) actually appear in the gold "
        "paragraph's text; this is exactly the lexical overlap BM25's score is built from, so "
        "an empty value here explains a bad `rank_bm25_only` directly. `bm25_actual_top1_para_num` "
        "/ `bm25_matched_terms_at_its_top1` show the same thing for whichever paragraph BM25 "
        "*actually* ranked #1 (useful when that's a different, wrong paragraph — you can see "
        "exactly which coincidental words pulled it to the top). These columns are identical "
        "across every model/encoding row for a given query, since BM25 never touches an "
        "embedding at all. `hit@k_*` columns are blank (NaN) rather than counted as a miss when "
        "`gold_paragraph_found_in_parse` is False — see the coverage figures in every aggregate "
        "table for how many rows this affects."
    )
    st.dataframe(results_df, use_container_width=True)

    st.markdown("### FAISS-only model leaderboard — which model performed better")
    st.caption(
        "Ranked by FAISS-only Accuracy@1, tie-broken by MRR, across every query in the "
        "uploaded CSV (all judgments combined) whose gold paragraph was locatable in the parse "
        "— see `coverage` for how many that was out of the total uploaded. This isolates the "
        "embedding-model comparison from BM25 entirely — exactly the '(0,1) pure semantic' "
        "condition."
    )
    faiss_leaderboard = results_df.groupby(["model", "encoding"]).agg(
        n_queries=("query_id", "count"),
        n_gold_found=("gold_paragraph_found_in_parse", "sum"),
        coverage=("gold_paragraph_found_in_parse", "mean"),
        faiss_acc1=("hit@1_faiss", "mean"), faiss_acc3=("hit@3_faiss", "mean"),
        faiss_acc5=("hit@5_faiss", "mean"), faiss_acc10=("hit@10_faiss", "mean"),
        faiss_mrr=("rr_faiss", "mean"),
    ).round(4).reset_index().sort_values(
        ["faiss_acc1", "faiss_mrr"], ascending=[False, False]
    ).reset_index(drop=True)
    faiss_leaderboard.insert(0, "rank", faiss_leaderboard.index + 1)
    st.dataframe(faiss_leaderboard, use_container_width=True)
    if not faiss_leaderboard.empty:
        top = faiss_leaderboard.iloc[0]
        st.success(
            f"🏆 Best FAISS-only combination: **{top['model']} — {top['encoding']}** "
            f"(Acc@1={top['faiss_acc1']:.3f}, Acc@5={top['faiss_acc5']:.3f}, MRR={top['faiss_mrr']:.3f}, "
            f"coverage={top['coverage']:.0%} of {int(top['n_queries'])} queries)."
        )

    st.markdown("### Aggregate summary (grouped by judgment × model × encoding)")
    agg = results_df.groupby(["judgment_id", "model", "encoding"]).agg(
        n_queries=("query_id", "count"),
        n_gold_found=("gold_paragraph_found_in_parse", "sum"),
        coverage=("gold_paragraph_found_in_parse", "mean"),
        faiss_acc1=("hit@1_faiss", "mean"), faiss_acc5=("hit@5_faiss", "mean"),
        faiss_acc10=("hit@10_faiss", "mean"), faiss_mrr=("rr_faiss", "mean"),
        bm25_acc1=("hit@1_bm25", "mean"), bm25_acc5=("hit@5_bm25", "mean"),
        bm25_acc10=("hit@10_bm25", "mean"), bm25_mrr=("rr_bm25", "mean"),
    ).round(4).reset_index()
    st.dataframe(agg, use_container_width=True)

    if (results_df["query_style"].astype(str) != "").any():
        st.markdown("### Aggregate by query style (bare_keyword / natural_question / fact_pattern / paraphrase)")
        agg_style = results_df.groupby(["query_style"]).agg(
            n_queries=("query_id", "count"),
            n_gold_found=("gold_paragraph_found_in_parse", "sum"),
            coverage=("gold_paragraph_found_in_parse", "mean"),
            faiss_acc1=("hit@1_faiss", "mean"), faiss_mrr=("rr_faiss", "mean"),
            bm25_acc1=("hit@1_bm25", "mean"), bm25_mrr=("rr_bm25", "mean"),
        ).round(4).reset_index()
        st.dataframe(agg_style, use_container_width=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download per-query results CSV", results_df.to_csv(index=False),
                            "batch_eval_per_query_results.csv", "text/csv", key="batch_perquery_download_btn")
    with c2:
        st.download_button("Download aggregate summary CSV", agg.to_csv(index=False),
                            "batch_eval_aggregate_summary.csv", "text/csv", key="batch_agg_download_btn")
    with c3:
        st.download_button("Download FAISS-only leaderboard CSV", faiss_leaderboard.to_csv(index=False),
                            "batch_eval_faiss_only_leaderboard.csv", "text/csv", key="batch_leaderboard_download_btn")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Evidence Extraction", page_icon="🔎", layout="wide")
    st.title("🔎 Evidence Extraction")
    st.caption(
        "Interactively query one judgment, or batch-evaluate a query test-set CSV against "
        "one or more judgments — measuring where the gold paragraph lands under FAISS-only "
        "and BM25-only retrieval, for every model/encoding combination."
    )

    mode = st.radio(
        "Mode",
        ["Single judgment — interactive query", "Batch CSV evaluation — test-set mode"],
        horizontal=True,
    )
    st.markdown("---")

    if mode.startswith("Batch"):
        run_batch_mode()
    else:
        run_single_query_mode()


if __name__ == "__main__":
    main()