"""
multi_judgment_app.py

Interactive hybrid retrieval across several judgments at once. A variant of
controlled_eval_app.py's single-judgment mode: any number of PDF/TXT/JSON
judgments can be uploaded, their paragraphs are pooled into one candidate set,
and each query is ranked against all of them. Every result shows the judgment
it came from. Used for the multi-judgment case study (Section 5.4).

Run:
    uv run streamlit run evaluation/multi_judgment_app.py

Paragraph splitting, role tagging, model loading, the two inference encodings
and the relevance gate work as in controlled_eval_app.py (see its docstring).
The gate is applied to the pooled set, so a query with nothing relevant in any
uploaded judgment is reported as unconfirmed. Embedding caches are keyed on
the names and sizes of all uploaded files, so they are rebuilt whenever the
set of judgments changes.

Note: this app runs InLegalBERT with a 256-token window; controlled_eval_app.py
uses its full 512 tokens.

Requires: streamlit, nltk, pdfplumber (PDF input only); `uv sync --extra eval`.
"""

import os
# Must be set BEFORE any tokenizer is loaded. Stops the Rust `tokenizers`
# backend from spawning its own worker threads (see controlled_eval_app.py).
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

EMBED_MODELS = {
    "MiniLM (general)":            "sentence-transformers/all-MiniLM-L6-v2",
    "Swiss Legal SBERT (domain)":  "Stern5497/sBert-swiss-legal-base",
    "InLegalBERT (Indian domain, no SBERT tuning)": "law-ai/InLegalBERT",
    "Mine_Finetuned_Indian_Legal": os.environ.get("LEGACY_MODEL_PATH", str(REPO_ROOT / "data" / "models" / "sbert-legal-v1-legacy")),
    "Mine_Finetuned_Indian_Legal_ChunkPool": os.environ.get("CHUNKPOOL_MODEL_PATH", str(REPO_ROOT / "data" / "models" / "sbert-legal-v1-chunkpoolmean")),
}
# InLegalBERT is a raw HF BERT checkpoint with no built-in SBERT pooling
# head, so it needs the manual Transformer+Pooling wrapper in get_embedder()
# below. Both Mine_Finetuned_* entries are self-contained sentence-
# transformers model folders (saved via .save()) and do NOT need this.
NEEDS_MANUAL_POOLING = {"law-ai/InLegalBERT"}
MANUAL_POOLING_MAX_SEQ_LEN = 256
MANUAL_POOLING_MODE = "mean"

WEIGHT_GRID = [(0.0,1.0), (0.3,0.7), (0.5,0.5), (0.7,0.3), (1.0,0.0)]

# ── Chunk-pool encoding (shared with training; see src/common/chunk_pool_encoding.py) ──
CHUNK_POOL_SRC_PATH = os.environ.get("CHUNK_POOL_SRC_PATH", str(REPO_ROOT / "src"))
if CHUNK_POOL_SRC_PATH not in sys.path:
    sys.path.insert(0, CHUNK_POOL_SRC_PATH)
from common.chunk_pool_encoding import encode_with_chunk_pool
import torch

ENCODING_MODES = {
    "Legacy (truncated)": None,
    "Chunk-Pool Mean":    "mean",
}
CHUNK_POOL_BATCH_SIZE = 8  # mini-batched to avoid the per-batch chunk-count
                           # memory blowup found earlier during ablation training/eval

DEFAULT_COSINE_MIN_BY_MODEL = {
    "MiniLM (general)": 0.35,
    "Swiss Legal SBERT (domain)": 0.35,
    "InLegalBERT (Indian domain, no SBERT tuning)": 0.55,   # unvalidated for this checkpoint
    "Mine_Finetuned_Indian_Legal": 0.35,
    "Mine_Finetuned_Indian_Legal_ChunkPool": 0.35,   # carried over from the legacy fine-tuned model's floor
}
DEFAULT_COSINE_MIN_FALLBACK = 0.35     # used for any model label not listed above
DEFAULT_BM25_MIN_RAW_THRESHOLD = 3.0   # raw rank_bm25 Okapi score (pre-minmax) — shared across models
DEFAULT_SCORE_GAP_MIN = 0.03           # combined-score gap between rank 1 and rank 2
DEFAULT_MIN_TERM_OVERLAP = 2           # min distinct query terms that must appear in the paragraph
                                        # for the BM25 branch to count

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL LOADING (as in controlled_eval_app.py)
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

@st.cache_resource(show_spinner=False)
def get_embedder(model_name):
    from sentence_transformers import SentenceTransformer
    if model_name in NEEDS_MANUAL_POOLING:
        Transformer, Pooling = _load_pooling_classes()
        word_embedding_model = Transformer(model_name, max_seq_length=MANUAL_POOLING_MAX_SEQ_LEN)
        pooling_model = Pooling(_embedding_dim(word_embedding_model), pooling_mode=MANUAL_POOLING_MODE)
        return SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
    return SentenceTransformer(model_name, device="cpu")

@torch.no_grad()
def encode_texts(model_name, texts, encoding_mode_label, legacy_batch_size=32):
    embedder = get_embedder(model_name)
    texts = list(texts)
    pooling = ENCODING_MODES[encoding_mode_label]

    with _ENCODE_LOCK:
        if pooling is None:
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
# PARAGRAPH SPLITTING (as in controlled_eval_app.py)
# ══════════════════════════════════════════════════════════════════════════════

def split_json_judgment(data, prefix=""):
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

def _nratio(a, b):
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
        r'\(?\d{4}\)?\s+\d+\s+(SCC|SCR|AIR|All|Bom|Mad|Cal)\b',
        r'\breported\s+(as|in)\b',
        r'\bAIR\s?\d{4}\b',
        r'\bthis\s+court\s+(has\s+)?held\s+in\b',
        r'\bin\s+the\s+case\s+of\b',
        r'\breliance\s+(is\s+)?placed\s+on\b',
        r'\bthe\s+law\s+(laid\s+down|settled)\b',
        r'\bprecedent\b',
        r'\bratio\s+decidendi\b',
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

    if _REJECTION_CUES.search(text):
        for role in ('ARG_P', 'ARG_R'):
            if scores[role] > 0:
                scores['REASON'] += scores[role]
                scores[role] = 0

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
    segments = [{'text': p['text'], 'role': p['rhetorical_role']} for p in paragraphs]
    return segments, n, len(paragraphs)

def extract_txt_segments(text):
    full_text = re.sub(r'\s+', ' ', text).strip()
    boundary = _find_para_boundary(full_text)
    body_text = full_text[boundary:] if boundary is not None else full_text
    paragraphs = _role_tag_paragraphs(_split_paragraphs_core(body_text))
    segments = [{'text': p['text'], 'role': p['rhetorical_role']} for p in paragraphs]
    return segments, len(paragraphs)

@st.cache_data(show_spinner=False)
def parse_uploaded_judgment_cached(file_bytes, file_name):
    """Returns (paragraphs list, info string, roles list-or-None) for ONE
    file, based on its extension. Cached on (file_bytes, file_name) so the
    LIS split + role-tagging pipeline runs once per uploaded file, not on
    every Streamlit rerun."""
    name = file_name.lower()

    if name.endswith(".json"):
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        paragraphs = split_json_judgment(data)
        n_top = len(data) if isinstance(data, dict) else None
        info = f"JSON — {n_top} top-level fields -> {len(paragraphs)} paragraphs (recursive field-path split)."
        return paragraphs, info, None

    elif name.endswith(".pdf"):
        segments, n_pages, n_raw_paras = extract_pdf_segments(file_bytes)
        paragraphs = [s["text"] for s in segments]
        roles = [s["role"] for s in segments]
        info = f"PDF — {n_pages} pages -> {len(paragraphs)} paragraphs (one per real judgment paragraph, no merging)."
        return paragraphs, info, roles

    else:  # .txt or unspecified
        text = file_bytes.decode("utf-8", errors="replace")
        segments, n_raw_paras = extract_txt_segments(text)
        paragraphs = [s["text"] for s in segments]
        roles = [s["role"] for s in segments]
        info = f"TXT — {len(paragraphs)} paragraphs (one per real judgment paragraph, no merging)."
        return paragraphs, info, roles

def parse_uploaded_judgment(uploaded_file):
    """Thin wrapper — extracts (bytes, name) so the cached function above
    has a stable, hashable cache key across Streamlit reruns."""
    return parse_uploaded_judgment_cached(uploaded_file.getvalue(), uploaded_file.name)

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-JUDGMENT: parse + pool every uploaded file into one combined corpus
# ══════════════════════════════════════════════════════════════════════════════

def _unique_judgment_id(file_name, seen_counts):
    """Two uploads can share a filename (e.g. two files both named
    '239.pdf' picked from different folders) — disambiguate with a
    trailing ' (#2)' suffix rather than silently colliding two different
    judgments under one label."""
    seen_counts[file_name] = seen_counts.get(file_name, 0) + 1
    n = seen_counts[file_name]
    return file_name if n == 1 else f"{file_name} (#{n})"

def parse_all_uploaded_judgments(uploaded_files):
    """Parses every uploaded file independently (each still cached per-file
    via parse_uploaded_judgment_cached), then pools all paragraphs into
    flat, PARALLEL lists:

      all_paragraphs   — paragraph text
      all_roles        — rhetorical role, or None if that file's format
                          doesn't produce roles (JSON) — 'None' role is
                          rendered as no tag, same as the single-doc version
      all_judgment_ids — which uploaded file (disambiguated name) each
                          paragraph came from — THE key addition that makes
                          multi-judgment retrieval traceable back to source

    Also returns per_judgment_info: an ordered dict of
      judgment_id -> {"info": str, "n_paragraphs": int, "error": str|None}
    for the Step 2 sanity-check display, and a flat list of any files that
    failed to parse (shown as warnings, not fatal — the rest still work).
    """
    all_paragraphs, all_roles, all_judgment_ids = [], [], []
    per_judgment_info = {}
    seen_counts = {}
    parse_errors = []

    for f in uploaded_files:
        judgment_id = _unique_judgment_id(f.name, seen_counts)
        try:
            paragraphs, info, roles = parse_uploaded_judgment(f)
        except Exception as e:
            parse_errors.append((judgment_id, str(e)))
            per_judgment_info[judgment_id] = {"info": None, "n_paragraphs": 0, "error": str(e)}
            continue

        per_judgment_info[judgment_id] = {"info": info, "n_paragraphs": len(paragraphs), "error": None}
        all_paragraphs.extend(paragraphs)
        all_judgment_ids.extend([judgment_id] * len(paragraphs))
        if roles is None:
            all_roles.extend([None] * len(paragraphs))
        else:
            all_roles.extend(roles)

    # If EVERY paragraph's role is None (e.g. all-JSON uploads), keep roles
    # as None overall so downstream role-tag rendering matches the
    # single-doc version's behavior exactly. Otherwise keep per-paragraph
    # None entries as-is (rendered as no role tag for just that paragraph).
    roles_out = None if all(r is None for r in all_roles) else all_roles

    return all_paragraphs, roles_out, all_judgment_ids, per_judgment_info, parse_errors

# ══════════════════════════════════════════════════════════════════════════════
# SCORING (unchanged mechanically — now just operates over the pooled corpus)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_paragraph_embeddings(model_name, corpus_key, paragraphs_tuple, encoding_mode_label):
    """Cache key includes corpus_key (derived from the FULL uploaded file
    set, see _corpus_key() below) and encoding_mode_label, so embeddings
    for a given model+encoding are only recomputed when the uploaded set of
    judgments actually changes."""
    return encode_texts(model_name, paragraphs_tuple, encoding_mode_label, legacy_batch_size=32)

def compute_raw_normalized_scores(query, paragraphs, bm25, model_name, para_embeddings, encoding_mode_label):
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
    bm25_ok = (bm25_score > 0.0 and bm25_score >= bm25_min
               and term_overlap >= min_term_overlap)
    cosine_ok = cosine_score > 0.0 and cosine_score >= cosine_min
    return cosine_ok or bm25_ok

def score_gap(combined):
    if len(combined) < 2:
        return None
    ordered = np.sort(combined)[::-1]
    return float(ordered[0] - ordered[1])

def _corpus_key(uploaded_files):
    """Stable cache key for the WHOLE uploaded set — name+size per file,
    sorted so upload order doesn't matter, joined into one string. Changes
    whenever any file is added, removed, or replaced; stays the same across
    reruns triggered by slider moves etc."""
    parts = sorted(f"{f.name}:{f.size}" for f in uploaded_files)
    return "|".join(parts)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Evidence Extraction (Multi-Judgment)", page_icon="🔎", layout="wide")
    st.title("🔎 Evidence Extraction — Multi-Judgment")
    st.caption(
        "Upload one or more judgments (PDF / TXT / JSON), split each into paragraphs, "
        "type a query, and see which paragraph — from WHICH judgment — each model and "
        "encoding mode ranks most relevant across the whole uploaded set."
    )

    st.markdown("### Step 1 — Upload judgment(s)")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or structured JSON — select multiple files to search across all of them",
        type=["pdf", "txt", "json"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Upload one or more files to continue. Supported: .pdf, .txt, .json"); return

    (paragraphs, roles, judgment_ids,
     per_judgment_info, parse_errors) = parse_all_uploaded_judgments(uploaded_files)

    for judgment_id, err in parse_errors:
        st.error(f"Failed to parse '{judgment_id}': {err}")
    if parse_errors:
        st.caption("If a failed file is a PDF: `pip install pdfplumber` may be needed. If JSON: confirm it's valid JSON.")

    if not paragraphs:
        st.error("No paragraphs were parsed from any uploaded file — nothing to search."); return

    n_ok = sum(1 for v in per_judgment_info.values() if v["error"] is None)
    st.success(f"Parsed {n_ok} of {len(uploaded_files)} uploaded judgment(s) -> {len(paragraphs)} paragraphs total.")
    if len(paragraphs) < 3:
        st.warning(
            f"Only {len(paragraphs)} paragraph(s) parsed across all uploads — the splitter may not have found "
            f"the structure it expected. Check Step 2 below."
        )

    st.markdown("### Step 2 — Parsed paragraphs, grouped by judgment (sanity check before querying)")
    for judgment_id, meta in per_judgment_info.items():
        if meta["error"] is not None:
            continue
        with st.expander(f"📄 {judgment_id} — {meta['n_paragraphs']} paragraphs — {meta['info']}",
                          expanded=meta["n_paragraphs"] < 3):
            # Slice this judgment's paragraphs back out of the pooled list —
            # cheap (uploaded sets are small) and avoids threading a second
            # per-judgment structure through parsing just for display.
            idxs = [i for i, jid in enumerate(judgment_ids) if jid == judgment_id]
            for local_rank, i in enumerate(idxs, 1):
                role_tag = f" `{roles[i]}`" if roles and roles[i] else ""
                st.markdown(f"**[{local_rank}]**{role_tag} {paragraphs[i][:400]}{'…' if len(paragraphs[i]) > 400 else ''}")

    st.markdown("### Step 3 — Model(s), encoding mode(s), & weight")
    st.caption(
        "Pick one or more base models AND one or more encoding modes — every combination "
        "is run and shown side by side, searched across ALL uploaded judgments pooled together. "
        "Chunk-Pool Mean chunks any paragraph exceeding the model's token budget and mean-pools "
        "the chunk embeddings, same as training/eval; Legacy truncates at the token budget like before."
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        _default_models = [k for k in EMBED_MODELS if k.startswith("Mine_Finetuned")]
        selected_models = st.multiselect(
            "Base embedding model(s) to compare", list(EMBED_MODELS.keys()),
            default=_default_models or [list(EMBED_MODELS.keys())[-1]],
        )
        selected_modes = st.multiselect(
            "Encoding mode(s) at inference", list(ENCODING_MODES.keys()),
            default=list(ENCODING_MODES.keys()),
        )
    with col2:
        bm25_weight = st.slider("BM25 weight (FAISS weight = 1 − this)", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"BM25={bm25_weight:.2f} · FAISS={1-bm25_weight:.2f}")

    if not selected_models or not selected_modes:
        st.warning("Select at least one model and one encoding mode to continue.")
        return

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
            "mode) combination**, not just per model. BM25's floor stays shared, since BM25 "
            "runs on the same token index regardless of model or encoding mode. This gate now "
            "applies across the FULL pooled multi-judgment set — a query with nothing relevant "
            "in any uploaded judgment will fail the gate against the whole pool, not just one document."
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
                     "coincidental word — requires this many distinct query terms to actually appear."
            )

    st.markdown("### Step 4 — Query")
    query = st.text_input("Enter your search query", placeholder="e.g. grounds for issue estoppel in non-criminal proceedings")
    show_grid = st.checkbox("Also show top-1 paragraph's rank across the full 5-point weight grid (robustness check)", value=False)
    run = st.button("Search", type="primary", disabled=not query.strip())

    if not run:
        return

    corpus_key = _corpus_key(uploaded_files)
    bm25 = BM25Okapi([preprocess(p) for p in paragraphs])

    st.markdown("---")
    st.warning(
        f"### 📍 Results below are computed at: **BM25 weight = {bm25_weight:.2f}, "
        f"FAISS weight = {1 - bm25_weight:.2f}**, searched across **{n_ok} judgment(s)**, "
        f"**{len(paragraphs)} paragraphs total**.\n"
        f"Change the slider in Step 3 and click Search again to see results at a "
        f"different weight — nothing below updates automatically when you move the slider."
    )

    all_download_rows = []

    for combined_label, model_name, encoding_mode_label, base_label in models_to_run:
        st.markdown("---")
        st.subheader(f"{combined_label}  ·  weight (BM25={bm25_weight:.2f}, FAISS={1-bm25_weight:.2f})")
        with st.spinner(f"Encoding {len(paragraphs)} paragraphs across {n_ok} judgment(s) with {combined_label}…"):
            para_emb = build_paragraph_embeddings(model_name, corpus_key, tuple(paragraphs), encoding_mode_label)
        bm25_norm, faiss_norm, bm25_raw, faiss_raw = compute_raw_normalized_scores(
            query, paragraphs, bm25, model_name, para_emb, encoding_mode_label
        )
        order, combined = combine_and_rank(bm25_norm, faiss_norm, bm25_weight)

        top_idx = order[0]
        role_tag = f" · role: `{roles[top_idx]}`" if roles and roles[top_idx] else ""
        judgment_tag = f" · **from: {judgment_ids[top_idx]}**"

        combo_cosine_min = cosine_min_by_combo.get(combined_label, DEFAULT_COSINE_MIN_FALLBACK)
        top_term_overlap = len(set(preprocess(query)) & set(preprocess(paragraphs[top_idx])))
        gate_ok = passes_relevance_gate(
            float(bm25_raw[top_idx]), float(faiss_raw[top_idx]), top_term_overlap,
            combo_cosine_min, bm25_min, min_term_overlap
        )
        gap = score_gap(combined)

        if not gate_ok:
            st.error(
                "⚠️ **No paragraph clearly matches this query.** Neither the best raw cosine "
                "similarity nor the best raw BM25 score clears its floor, across ANY of the "
                f"{n_ok} uploaded judgment(s) — none of them may address what was asked. The "
                "closest match is shown below for reference only; treat it as unconfirmed."
            )
        elif gap is not None and gap < gap_min:
            st.caption(
                f"ℹ️ Top-1 and top-2 scores are close (gap={gap:.3f}) — no single paragraph "
                f"stands out clearly (they may even be from different judgments); worth "
                f"checking the next candidates below too."
            )

        st.markdown(f"**Most relevant paragraph (rank 1 of {len(paragraphs)}) "
                     f"at BM25={bm25_weight:.2f}/FAISS={1-bm25_weight:.2f}{role_tag}{judgment_tag}:**")
        st.info(paragraphs[top_idx])
        st.caption(
            f"combined={combined[top_idx]:.4f} · bm25_norm={bm25_norm[top_idx]:.4f} · "
            f"faiss_norm={faiss_norm[top_idx]:.4f}  ‖  "
            f"raw: bm25={bm25_raw[top_idx]:.3f} (floor {bm25_min:.1f}, needs ≥{min_term_overlap} matched terms — "
            f"this paragraph matched {top_term_overlap}) · "
            f"cosine={faiss_raw[top_idx]:.3f} (floor {combo_cosine_min:.2f}, this model+encoding) · "
            f"gate={'✅ passed' if gate_ok else '❌ not met'}"
        )

        with st.expander(f"Next {min(9, len(order)-1)} candidates (across all uploaded judgments)"):
            for rank, idx in enumerate(order[1:10], start=2):
                jtag = f" `{judgment_ids[idx]}`"
                st.markdown(f"**[{rank}]**{jtag} (combined={combined[idx]:.4f}) {paragraphs[idx][:300]}"
                            f"{'…' if len(paragraphs[idx]) > 300 else ''}")

        for rank, idx in enumerate(order[:10], start=1):
            all_download_rows.append({
                "Query": query,
                "Model": base_label, "Encoding": encoding_mode_label,
                "BM25 weight": bm25_weight, "FAISS weight": round(1 - bm25_weight, 2),
                "Rank": rank,
                "Judgment": judgment_ids[idx],
                "Paragraph # (within judgment)": (
                    [i for i, jid in enumerate(judgment_ids) if jid == judgment_ids[idx]].index(idx) + 1
                ),
                "Role": (roles[idx] if roles and roles[idx] else ""),
                "Combined score": round(float(combined[idx]), 4),
                "BM25 norm": round(float(bm25_norm[idx]), 4), "FAISS norm": round(float(faiss_norm[idx]), 4),
                "BM25 raw": round(float(bm25_raw[idx]), 4), "FAISS raw (cosine)": round(float(faiss_raw[idx]), 4),
                "Relevance gate passed (top-1 only)": gate_ok if rank == 1 else "",
                "Top1/Top2 score gap": round(gap, 4) if (rank == 1 and gap is not None) else "",
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
    query_slug = re.sub(r"[^a-zA-Z0-9]+", "_", query.strip())[:40].strip("_")
    st.download_button(
        f"Download top-10-per-combo CSV — query='{query[:40]}', weight=({bm25_weight:.2f},{1-bm25_weight:.2f}), "
        f"{n_ok} judgment(s)",
        download_df.to_csv(index=False),
        f"evidence_extraction_multi_{query_slug}_bm25_{bm25_weight:.2f}.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()