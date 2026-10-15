"""
fire_aila_eval_app.py

FIRE/AILA 2019 judgment retrieval (Section 5.2 of the paper). Ranks a pool of
judgments for each AILA query with BM25, a sentence-embedding model, or a
weighted combination of the two, and reports TREC metrics (MAP, P@10, BPREF,
MRR). This app produced the files in results/fire_aila/; Tables 3 and 4 use the
Facts formulation.

Run:
    uv run streamlit run evaluation/fire_aila_eval_app.py

What is varied
--------------
- Query formulation: Facts (the AILA query text), Issue 1/2/3 (LLM-generated
  issue statements, pre-computed), the three issues joined, and Reciprocal
  Rank Fusion (k = 60) of Facts and the three issues.
- BM25 weight alpha in {0, 0.3, 0.5, 0.7, 1}: score = alpha * BM25 +
  (1 - alpha) * cosine, both min-max normalised over the pool for each query.
- Model: untuned all-MiniLM-L6-v2, Swiss Legal SBERT, InLegalBERT (mean pooling
  over a 256-token window, no contrastive tuning), and our Legacy-trained and
  CP-trained checkpoints.
- Inference encoding: legacy (truncated), chunk-pool mean or chunk-pool max,
  applied to any model, so that the encoding and the fine-tuning can be
  separated.

BM25 and the document embeddings are computed once per (query text, model,
encoding); every weight is then a cheap recombination of the same two score
arrays. Rankings for RRF are fused at each weight from the already-combined
Facts/Issue rankings.

Inputs
------
- AILA query file, one line per query: "AILA_Q1||<query text>"
- JSON of pre-generated issue statements per query (Issue_1..Issue_3)
- A folder of judgment texts (TXT or JSON), file names containing the case ID
  (e.g. C14.txt). Use the 186 judgments in data/fire_aila/pool_186_judgment_ids.txt
  or the full 2,914-judgment collection.
- AILA 2019 relevance judgments as JSON, {"AILA_Q1": ["C14", "C9"], ...}, at
  data/fire_aila/aila2019_qrels.json or the path in AILA_QRELS. The AILA data
  is not redistributed here; it is available from the track organisers.

Outputs (download buttons): embedding_variant_weight_summary.csv (MAP, P@10,
BPREF, MRR per configuration) and gt_hit_rank_detail.csv (rank of every
relevant judgment per query and configuration).

Implementation notes: encoding runs under torch.no_grad(), since chunk-pooling
long judgments otherwise accumulates autograd graphs and exhausts memory. The
cached model is shared across Streamlit sessions, so encoding calls are
serialised behind a lock and TOKENIZERS_PARALLELISM is disabled. Results are
rendered in a Streamlit fragment so that download buttons do not rerun the
retrieval. A query formulation with empty text yields an empty ranking.

Requires: streamlit, nltk (`uv sync --extra eval`).
"""

import os
# Must be set BEFORE any tokenizer is loaded (i.e. before the first
# SentenceTransformer(...) call inside get_embedder()). Stops the Rust
# `tokenizers` backend from spawning its own internal worker threads --
# one of the two sources of the "Already borrowed" race.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
import re, json, glob, sys, gc, threading
import pandas as pd
import numpy as np
from collections import defaultdict
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

# ══════════════════════════════════════════════════════════════════════════════
# THREAD-SAFETY: single global lock serializing all tokenizer/model encode
# calls. @st.cache_resource shares one SentenceTransformer (and its fast
# tokenizer) across every session/thread on the server, and the tokenizer
# is not safe for concurrent use -- without this lock, two overlapping
# encode calls on the same model can raise `RuntimeError: Already borrowed`.
# ══════════════════════════════════════════════════════════════════════════════
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

_ENCODE_LOCK = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT GRID: (BM25 weight, dense weight)
# ══════════════════════════════════════════════════════════════════════════════

WEIGHT_GRID = [
    (0.0, 1.0),
    (0.3, 0.7),
    (0.5, 0.5),
    (0.7, 0.3),
    (1.0, 0.0),
]

EMBED_MODELS = {
    "MiniLM (general)":            "sentence-transformers/all-MiniLM-L6-v2",
    "Swiss Legal SBERT (domain)":  "Stern5497/sBert-swiss-legal-base",
    "InLegalBERT (Indian domain, no SBERT tuning)": "law-ai/InLegalBERT",
    "Mine_Finetuned_Indian_Legal": os.environ.get(
        "LEGACY_MODEL_PATH", str(REPO_ROOT / "data" / "models" / "sbert-legal-v1-legacy")),
    "Mine_Finetuned_Indian_Legal_ChunkPool": os.environ.get(
        "CHUNKPOOL_MODEL_PATH", str(REPO_ROOT / "data" / "models" / "sbert-legal-v1-chunkpoolmean")),
}

# InLegalBERT is a plain BERT checkpoint with no sentence-embedding pooling
# head, so get_embedder() wraps it with a mean-pooling layer.
NEEDS_MANUAL_POOLING = {
    "law-ai/InLegalBERT",
}
MANUAL_POOLING_MAX_SEQ_LEN = 256
MANUAL_POOLING_MODE = "mean"

# ── chunk-pool encoding support ──────────────────────────────────────────────
CHUNK_POOL_SRC_PATH = os.environ.get("CHUNK_POOL_SRC_PATH", str(REPO_ROOT / "src"))
if CHUNK_POOL_SRC_PATH not in sys.path:
    sys.path.insert(0, CHUNK_POOL_SRC_PATH)
from common.chunk_pool_encoding import encode_with_chunk_pool
import torch

# Encoding strategy is independent of which model you pick -- ANY model
# (even ones never chunk-pool fine-tuned) can be run through chunk-pool
# encoding at inference time.
ENCODING_MODES = {
    "Legacy (truncated)": None,
    "Chunk-Pool Mean":    "mean",
    "Chunk-Pool Max":     "max",
}
CHUNK_POOL_BATCH_SIZE = 8  # mini-batched to avoid the per-batch chunk-count
                           # memory blowup found earlier during ablation training/eval

# ══════════════════════════════════════════════════════════════════════════════
# STYLING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _style_diverging(val, vmin=-0.1, vmax=0.1):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 0:
        frac = min(v / vmax, 1.0) if vmax else 0.0
        r, g, b = 255 - int(frac*155), 255, 255 - int(frac*155)
    else:
        frac = min(v / vmin, 1.0) if vmin else 0.0
        r, g, b = 255, 255 - int(frac*155), 255 - int(frac*155)
    return f"background-color: rgb({r},{g},{b})"

def _style_sequential(val, vmin=0.0, vmax=0.5):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    v = max(min(v, vmax), vmin)
    frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    r, g, b = 255 - int(frac*175), 255, 255 - int(frac*175)
    return f"background-color: rgb({r},{g},{b})"

def _style_apply(styler, fn, subset):
    if hasattr(styler, "map"):
        return styler.map(fn, subset=subset)
    return styler.applymap(fn, subset=subset)

# ══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════════

# AILA 2019 relevance judgments, {"AILA_Q1": ["C14", "C9"], ...}. Not included
# in this repository; see the module docstring.
QRELS_PATH = os.environ.get("AILA_QRELS", str(REPO_ROOT / "data" / "fire_aila" / "aila2019_qrels.json"))


def load_qrels(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {qid: list(cases) for qid, cases in json.load(f).items()}
    except FileNotFoundError:
        return {}


GROUND_TRUTH_RAW = load_qrels(QRELS_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# TREC METRICS
# ══════════════════════════════════════════════════════════════════════════════

def average_precision(ranked, relevant):
    if not relevant: return 0.0
    hits, ap = 0, 0.0
    for i, cid in enumerate(ranked, 1):
        if cid in relevant:
            hits += 1; ap += hits / i
    return ap / len(relevant)

def precision_at_k(ranked, relevant, k=10):
    return len(set(ranked[:k]) & relevant) / k if ranked else 0.0

def reciprocal_rank(ranked, relevant):
    for i, cid in enumerate(ranked, 1):
        if cid in relevant: return 1.0 / i
    return 0.0

def bpref_score(ranked, relevant, all_ids):
    R = len(relevant)
    if R == 0: return 0.0
    non_rel = all_ids - relevant
    cap     = min(R, len(non_rel)) if non_rel else 1
    score, nrs = 0.0, 0
    for cid in ranked:
        if cid in non_rel: nrs += 1
        elif cid in relevant:
            score += 1.0 - (min(nrs, cap) / cap)
    return score / R

def trec_metrics(ranked, relevant, all_ids):
    ranked = [c for c in ranked if c]
    ranks = {}
    for i, cid in enumerate(ranked, 1):
        if cid in relevant and cid not in ranks:
            ranks[cid] = i
    return {
        "AP":     average_precision(ranked, relevant),
        "P@10":   precision_at_k(ranked, relevant, 10),
        "BPREF":  bpref_score(ranked, relevant, all_ids),
        "RR":     reciprocal_rank(ranked, relevant),
        "hits":   sorted(set(ranked) & relevant),
        "ranks":  ranks,
        "missed": sorted(relevant - set(ranked)),
    }

def aggregate(per_query):
    labels = {"AP":"MAP","P@10":"P@10","BPREF":"BPREF","RR":"MRR"}
    return {labels[k]: float(np.mean([v[k] for v in per_query.values()])) for k in labels}

def paired_significance(per_query_a, per_query_b, metric="AP", n_boot=10000, seed=42):
    """Paired bootstrap significance test between two per-query result sets."""
    qids = sorted(set(per_query_a.keys()) & set(per_query_b.keys()))
    if not qids:
        return 0.0, 1.0
    a = np.array([per_query_a[q][metric] for q in qids])
    b = np.array([per_query_b[q][metric] for q in qids])
    diffs = a - b
    observed = diffs.mean()
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diffs[idx].mean()
    p = (np.mean(boot_means <= 0) if observed >= 0 else np.mean(boot_means >= 0)) * 2
    return float(observed), float(min(p, 1.0))

# ══════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL LOADING (with InLegalBERT manual-pooling wrapper)
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
    # NOTE: this object is cached and SHARED across every Streamlit session/
    # thread on the server for the lifetime of the process. Its tokenizer is
    # not safe for concurrent use -- callers must go through _ENCODE_LOCK
    # (see encode_texts below), not call .encode()/.tokenize() on it directly.
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
    Universal encoder used for BOTH documents and queries. encoding_mode_label
    is a key into ENCODING_MODES, e.g. "Legacy (truncated)", "Chunk-Pool Mean",
    or "Chunk-Pool Max". Any model can be run under any mode -- this is what
    lets you separate "does the encoding method matter" from "does the
    fine-tuning matter".

    Runs under torch.no_grad(): without it, chunk-pooling long judgments
    keeps an autograd graph per document and runs out of memory. Model calls
    are serialised behind _ENCODE_LOCK because the cached model and its
    tokenizer are shared across Streamlit sessions.
    """
    embedder = get_embedder(model_name)
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

@st.cache_resource(show_spinner=False)
def build_doc_embeddings(model_name, encoding_mode_label, corpus_key, doc_texts):
    return encode_texts(model_name, doc_texts, encoding_mode_label)

def _minmax(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-9)

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-PASS RAW SCORES + WEIGHT-GRID COMBINATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_raw_normalized_scores(query, doc_texts, doc_ids, bm25, model_name, encoding_mode_label, doc_embeddings):
    """
    The expensive half, computed exactly ONCE per (query text, model,
    encoding mode): BM25 lookup + sentence-transformer encode (legacy or
    chunk-pool). Returns (bm25_norm, faiss_norm), both min-max normalized
    to [0,1] -- ready to be combined with ANY weight pair via
    combine_scores() without recomputing either.
    """
    n = len(doc_texts)
    q_toks   = preprocess(query)
    bm25_raw = np.array(bm25.get_scores(q_toks)) if q_toks else np.zeros(n)

    if query.strip():
        q_emb = encode_texts(model_name, [query], encoding_mode_label)[0]
        faiss_raw = doc_embeddings @ q_emb
    else:
        faiss_raw = np.zeros(n)

    return _minmax(bm25_raw), _minmax(faiss_raw)

def combine_scores(bm25_norm, faiss_norm, bm25_weight, doc_ids):
    """Cheap step — the only thing that changes per weight-grid point."""
    combined = bm25_weight * bm25_norm + (1.0 - bm25_weight) * faiss_norm
    order = np.argsort(combined)[::-1]
    return [doc_ids[i] for i in order]

# ══════════════════════════════════════════════════════════════════════════════
# RECIPROCAL RANK FUSION
# ══════════════════════════════════════════════════════════════════════════════

def rrf(ranked_lists, k=60):
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)

def clean_issue(text):
    return re.sub(r"---END---.*", "", text or "").strip()

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT & QUERY LOADING
# ══════════════════════════════════════════════════════════════════════════════

def cid_from_filename(fname):
    m = re.search(r"(C\d+)", os.path.basename(fname), re.IGNORECASE)
    return m.group(1).upper() if m else ""

def load_txt_doc(fp):
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    cid = cid_from_filename(fp)
    return {"case_id": cid, "text": text, "filename": os.path.basename(fp), "format": "TXT"}

def load_json_doc(fp):
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        return None
    cid = cid_from_filename(fp)
    parts = []
    for k, v in raw.items():
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")
        elif isinstance(v, list):
            parts.append(f"{k}: {' '.join(str(x) for x in v)}")
    return {"case_id": cid, "text": "\n".join(parts),
            "filename": os.path.basename(fp), "format": "JSON"}

@st.cache_data
def load_corpus(folder):
    folder = folder.strip()
    if not os.path.isdir(folder): return []
    docs = []
    for fp in sorted(glob.glob(os.path.join(folder, "*.txt")) +
                     glob.glob(os.path.join(folder, "*.json"))):
        ext = os.path.splitext(fp)[1].lower()
        doc = load_txt_doc(fp) if ext == ".txt" else load_json_doc(fp)
        if doc and doc["case_id"]:
            docs.append(doc)
    return docs

def load_queries(fp):
    queries = {}
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "||" in line:
                qid, text = line.strip().split("||", 1)
                queries[qid.strip()] = text.strip()
    return queries

def load_issues(fp):
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        raw = json.load(f)
    out = {}
    for qid, row in raw.items():
        out[qid] = {
            "Issue_1": clean_issue(row.get("Issue_1", "")),
            "Issue_2": clean_issue(row.get("Issue_2", "")),
            "Issue_3": clean_issue(row.get("Issue_3", "")),
        }
    return out

# ══════════════════════════════════════════════════════════════════════════════
# CORE: RUN ONE (MODEL, ENCODING MODE) COMBO ACROSS ALL 6 VARIANTS × 5 WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = ["Facts", "Issue_1", "Issue_2", "Issue_3", "Issues_Combined", "RRF_All"]
BASE_VARIANTS = ["Facts", "Issue_1", "Issue_2", "Issue_3", "Issues_Combined"]  # RRF built from these

def run_embed_model_grid(model_name, encoding_mode_label, doc_texts, doc_ids, all_ids, bm25, doc_embeddings,
                          query_texts, issues_data, weight_grid=WEIGHT_GRID):
    """
    Returns pq[variant][weight_pair][qid] -> trec_metrics dict.

    For each query: encode/score Facts, Issue_1, Issue_2, Issue_3, and
    Issues_Combined ONCE EACH (5 encode calls total, not 5×len(weight_grid)),
    under whichever encoding_mode_label was selected for this combo. Then
    for every weight pair in weight_grid, combine those same 5 raw score
    arrays into rankings (cheap), and build RRF_All at that weight by
    fusing the Facts/Issue_1/Issue_2/Issue_3 rankings AT THAT WEIGHT — RRF
    itself needs no additional encode/BM25 call, it only reorders rankings
    that already exist.
    """
    pq = {v: {w: {} for w in weight_grid} for v in VARIANTS}

    for qid, facts in sorted(query_texts.items()):
        if not facts.strip():
            continue
        rel = set(GROUND_TRUTH_RAW.get(qid, []))
        iss = issues_data.get(qid, {})
        i1, i2, i3 = iss.get("Issue_1", ""), iss.get("Issue_2", ""), iss.get("Issue_3", "")
        combined_q = " | ".join(x for x in [i1, i2, i3] if x)

        # --- expensive part: once per variant text, not once per weight ---
        texts_by_variant = {
            "Facts": facts, "Issue_1": i1, "Issue_2": i2,
            "Issue_3": i3, "Issues_Combined": combined_q,
        }
        raw_by_variant = {}
        for v, text in texts_by_variant.items():
            if text.strip():
                raw_by_variant[v] = compute_raw_normalized_scores(
                    text, doc_texts, doc_ids, bm25, model_name, encoding_mode_label, doc_embeddings
                )
            else:
                # Empty query text gives an empty ranking at every weight.
                raw_by_variant[v] = None

        # --- cheap part: combine at every weight-grid point ---
        for (bw, fw) in weight_grid:
            ranked_by_variant = {}
            for v in BASE_VARIANTS:
                raw = raw_by_variant[v]
                ranked_by_variant[v] = combine_scores(raw[0], raw[1], bw, doc_ids) if raw else []

            ranked_by_variant["RRF_All"] = rrf([
                ranked_by_variant["Facts"], ranked_by_variant["Issue_1"],
                ranked_by_variant["Issue_2"], ranked_by_variant["Issue_3"],
            ])

            for v in VARIANTS:
                pq[v][(bw, fw)][qid] = trec_metrics(ranked_by_variant[v], rel, all_ids)

    return pq

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

VARIANT_LABELS = {
    "Facts":           "Facts only",
    "Issue_1":         "Issue 1 only",
    "Issue_2":         "Issue 2 only",
    "Issue_3":         "Issue 3 only",
    "Issues_Combined": "Issues combined (I1 | I2 | I3)",
    "RRF_All":         "RRF(Facts+I1+I2+I3)",
}

def _slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")

def display_single_grid(model_label, pq, weight_grid=WEIGHT_GRID):
    """
    One compact table per model/encoding combo: rows = variant, columns =
    MAP at each weight-grid point, plus the best weight for that variant
    highlighted. Full P@10/BPREF/MRR detail for a specific variant is
    available in the per-variant expander below this table.
    """
    slug = _slug(model_label)
    st.markdown(f"#### {model_label}")
    rows = []
    for v in VARIANTS:
        row = {"Variant": VARIANT_LABELS[v]}
        maps = {}
        for w in weight_grid:
            m = aggregate(pq[v][w])["MAP"]
            row[f"MAP@{w}"] = round(m, 4)
            maps[w] = m
        best_w = max(maps, key=maps.get)
        row["Best weight"] = f"{best_w}"
        row["Best MAP"] = round(maps[best_w], 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    map_cols = [f"MAP@{w}" for w in weight_grid]
    styled = df.style.format("{:.4f}", subset=map_cols + ["Best MAP"]) \
        .highlight_max(subset=map_cols, color="#d0f0e0")
    # Explicit key: this runs inside a loop over configurations, and
    # auto-generated widget ids can otherwise collide across reruns.
    st.dataframe(styled, use_container_width=True, hide_index=True, key=f"grid_main_{slug}")

    with st.expander(f"Full P@10 / BPREF / MRR per weight — {model_label}"):
        for v in VARIANTS:
            st.markdown(f"**{VARIANT_LABELS[v]}**")
            detail_rows = []
            for w in weight_grid:
                a = aggregate(pq[v][w])
                detail_rows.append({"Weight (bm25, faiss)": str(w),
                                     "MAP": round(a["MAP"],4), "P@10": round(a["P@10"],4),
                                     "BPREF": round(a["BPREF"],4), "MRR": round(a["MRR"],4)})
            ddf = pd.DataFrame(detail_rows)
            st.dataframe(ddf.style.format("{:.4f}", subset=["MAP","P@10","BPREF","MRR"]),
                         use_container_width=True, hide_index=True,
                         key=f"grid_detail_{slug}_{v}")
    return df

def display_perquery(model_label, pq_at_weight, qids, key_suffix, weight_label):
    with st.expander(f"Per-query AP breakdown — {model_label} @ weight {weight_label} (click to expand)"):
        rows = []
        for qid in qids:
            rel = set(GROUND_TRUTH_RAW.get(qid, []))
            row = {"Query": qid, "GT": len(rel)}
            for v in VARIANTS:
                row[VARIANT_LABELS[v]] = round(pq_at_weight[v][qid]["AP"], 4)
            ap_vals = {VARIANT_LABELS[v]: row[VARIANT_LABELS[v]] for v in VARIANTS}
            best_v  = max(ap_vals, key=ap_vals.get)
            row["Best variant"] = best_v if ap_vals[best_v] > 0 else "—"
            rows.append(row)
        df = pd.DataFrame(rows)
        ap_cols = [VARIANT_LABELS[v] for v in VARIANTS]
        styled = df.style.format("{:.4f}", subset=ap_cols)
        styled = _style_apply(styled, lambda v: _style_sequential(v, 0, 0.5), ap_cols)
        st.dataframe(styled, use_container_width=True, hide_index=True, key=f"perquery_{key_suffix}")
        st.download_button(f"Per-query CSV — {model_label} @ {weight_label}", df.to_csv(index=False),
                            f"perquery_{key_suffix}.csv", "text/csv", key=f"dl_{key_suffix}")

def display_query_detail(all_pq_at_weight, qids, weight_label):
    st.markdown(f"#### Ground truth retrieval detail — per query (weight = {weight_label})")
    st.caption("For a selected query: which GT case(s) were actually retrieved by each "
               "variant, and at what rank, at the weight chosen above. '—' means that GT "
               "case was not retrieved at all.")

    def _qlabel(qid):
        rel = GROUND_TRUTH_RAW.get(qid, [])
        return f"{qid}  ({len(rel)} GT doc{'s' if len(rel)!=1 else ''})"

    qid_sel = st.selectbox("Select query", qids, format_func=_qlabel, key="qid_detail_select")
    relevant = sorted(GROUND_TRUTH_RAW.get(qid_sel, []))
    if not relevant:
        st.info("No ground truth defined for this query."); return

    for label, pq in all_pq_at_weight.items():
        slug = _slug(label)
        st.markdown(f"**{label}**")
        rows = []
        for cid in relevant:
            row = {"GT Case": cid}
            for v in VARIANTS:
                rank = pq[v][qid_sel]["ranks"].get(cid)
                row[VARIANT_LABELS[v]] = rank if rank else "—"
            rows.append(row)
        df = pd.DataFrame(rows)
        ap_cols = [VARIANT_LABELS[v] for v in VARIANTS]

        def _rank_color(val):
            if val == "—" or pd.isna(val):
                return "background-color: rgb(255,225,225)"
            return "background-color: rgb(220,245,225)"

        styled = _style_apply(df.style, _rank_color, ap_cols)
        st.dataframe(styled, use_container_width=True, hide_index=True,
                     key=f"gtdetail_{slug}_{_slug(qid_sel)}")

        recap = []
        for v in VARIANTS:
            m = pq[v][qid_sel]
            recap.append(f"{VARIANT_LABELS[v]}: {len(m['hits'])}/{len(relevant)} found "
                         f"(AP={m['AP']:.3f})")
        st.caption(" · ".join(recap))

# A fragment, so that download buttons and selectboxes rerun only this
# function and not the retrieval above it.
@st.fragment
def display_results(all_pq, n_docs, n_queries, weight_grid=WEIGHT_GRID):
    """all_pq: dict[combined_label] -> pq[variant][weight][qid] -> metrics"""
    st.markdown("---")
    st.subheader(f"Results — {n_queries} queries · {n_docs} documents")
    st.caption(f"Every variant retrieved with BM25+FAISS, swept across weight grid {weight_grid}.")

    model_labels = list(all_pq.keys())

    for label in model_labels:
        display_single_grid(label, all_pq[label], weight_grid)

    st.markdown("---")
    st.markdown("### Detail view weight")
    detail_weight = st.selectbox(
        "Weight to use for per-query breakdown & ground-truth detail below",
        options=weight_grid, index=weight_grid.index((0.7, 0.3)) if (0.7,0.3) in weight_grid else 0,
        format_func=lambda w: f"BM25={w[0]}, FAISS={w[1]}",
        key="detail_weight_select",
    )
    weight_label = f"BM25={detail_weight[0]}, FAISS={detail_weight[1]}"
    all_pq_at_weight = {label: {v: all_pq[label][v][detail_weight] for v in VARIANTS}
                         for label in model_labels}
    qids = sorted(next(iter(all_pq_at_weight.values()))["Facts"].keys())

    if len(model_labels) == 2:
        st.markdown(f"#### MAP / P@10 / BPREF / MRR delta @ {weight_label} — (2nd combo − 1st combo)")
        m1, m2 = model_labels
        delta_rows = []
        for v in VARIANTS:
            a1 = aggregate(all_pq_at_weight[m1][v]); a2 = aggregate(all_pq_at_weight[m2][v])
            diff, p = paired_significance(all_pq_at_weight[m2][v], all_pq_at_weight[m1][v])
            delta_rows.append({
                "Variant": VARIANT_LABELS[v],
                "MAP Δ":   round(a2["MAP"]-a1["MAP"], 4),
                "P@10 Δ":  round(a2["P@10"]-a1["P@10"], 4),
                "BPREF Δ": round(a2["BPREF"]-a1["BPREF"], 4),
                "MRR Δ":   round(a2["MRR"]-a1["MRR"], 4),
                "p-value (paired bootstrap, AP)": round(p, 4),
            })
        ddf = pd.DataFrame(delta_rows)
        dcols = ["MAP Δ","P@10 Δ","BPREF Δ","MRR Δ"]
        styled_d = ddf.style.format({**{c: "{:+.4f}" for c in dcols}, "p-value (paired bootstrap, AP)": "{:.4f}"})
        styled_d = _style_apply(styled_d, lambda v: _style_diverging(v, -0.1, 0.1), dcols)
        st.dataframe(styled_d, use_container_width=True, hide_index=True, key="delta_table")
        st.caption(f"Positive (green) = **{m2}** scores higher · Negative (red) = **{m1}** scores higher. "
                   f"p < 0.05 suggests the AP difference is unlikely to be chance given {len(qids)} queries.")
    elif len(model_labels) > 2:
        st.caption(f"Pairwise delta/significance table only renders for exactly 2 combos at once "
                   f"— {len(model_labels)} selected. Use the per-combo grid tables above instead.")

    for label in model_labels:
        slug = _slug(label)
        display_perquery(label, all_pq_at_weight[label], qids, slug, weight_label)

    st.markdown("---")
    display_query_detail(all_pq_at_weight, qids, weight_label)

    st.markdown("#### Download full summary (all combos × variants × weights)")
    sum_rows = []
    for label in model_labels:
        for v in VARIANTS:
            for w in weight_grid:
                a = aggregate(all_pq[label][v][w])
                sum_rows.append({"Model x Encoding": label, "Variant": VARIANT_LABELS[v],
                                  "BM25 weight": w[0], "FAISS weight": w[1],
                                  "MAP": round(a["MAP"],4), "P@10": round(a["P@10"],4),
                                  "BPREF": round(a["BPREF"],4), "MRR": round(a["MRR"],4)})
    sum_df = pd.DataFrame(sum_rows)
    st.download_button("Summary CSV (model×encoding × variant × weight × aggregate)",
                        sum_df.to_csv(index=False), "embedding_variant_weight_summary.csv", "text/csv",
                        key="dl_summary_csv")

    st.markdown("#### Download full ground-truth hit/rank detail (all queries × variants × weights × combos)")
    flat_rows = []
    for label in model_labels:
        for w in weight_grid:
            for qid in qids:
                relevant_q = sorted(GROUND_TRUTH_RAW.get(qid, []))
                for v in VARIANTS:
                    m = all_pq[label][v][w][qid]
                    for cid in relevant_q:
                        flat_rows.append({
                            "Model x Encoding": label, "BM25 weight": w[0], "FAISS weight": w[1],
                            "Query": qid, "Variant": VARIANT_LABELS[v],
                            "GT_Case": cid, "Rank": m["ranks"].get(cid, "Not retrieved"),
                        })
    flat_df = pd.DataFrame(flat_rows)
    st.download_button("Full GT hit/rank detail CSV", flat_df.to_csv(index=False),
                        "gt_hit_rank_detail.csv", "text/csv", key="dl_gt_detail")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Issue Contribution × Weight Grid × Encoding", page_icon="⚖️", layout="wide")
    st.title("⚖️ AILA — Issue Contribution Analysis × BM25/FAISS Weight Grid × Encoding Mode")
    st.caption(
        f"Facts · Issue 1 · Issue 2 · Issue 3 · Issues combined · RRF(all 4) — "
        f"each swept across weight grid {WEIGHT_GRID} and encoding modes "
        f"{list(ENCODING_MODES.keys())}, single pass per (variant, encoding)."
    )

    st.sidebar.markdown(
        "### Variants per query\n"
        "**Facts** — raw query text only  \n"
        "**Issue 1/2/3** — LLM-pre-generated legal principles, queried separately  \n"
        "**Issues combined** — Issue_1 | Issue_2 | Issue_3 joined with ` | `  \n"
        "**RRF(all 4)** — Reciprocal Rank Fusion of Facts+I1+I2+I3, k=60, "
        "fused AT EACH weight grid point  \n\n"
        f"### Weight grid\n{WEIGHT_GRID}\n\n"
        "### Encoding modes\n"
        "**Legacy (truncated)** — plain model.encode(), silently truncates at "
        "max_seq_length (256 tokens for these models).  \n"
        "**Chunk-Pool Mean / Max** — chunk long documents, encode each chunk, "
        "then mean- or max-pool the chunk embeddings into one document vector. "
        "Applies to ANY model, not just chunk-pool fine-tuned ones — lets you "
        "separate encoding-method effects from fine-tuning effects.\n\n"
        "Raw BM25/FAISS scores are computed ONCE per query per variant text per "
        "encoding mode; all 5 weight points reuse those same scores (cheap "
        "recombination, not 5× the encode/BM25 cost).\n\n"
        "### No LLM at runtime\n"
        "Issues are pre-generated."
    )

    st.markdown("### Step 1 — Query file")
    query_path = st.text_input("Path to Query_doc.txt",
        value=os.environ.get("AILA_QUERIES", str(REPO_ROOT / "data" / "fire_aila" / "Query_doc.txt")))
    if not query_path.strip() or not os.path.isfile(query_path.strip()):
        st.error("Query file not found."); return
    query_texts = load_queries(query_path.strip())
    n_q = sum(1 for v in query_texts.values() if v.strip())
    if n_q == 0:
        st.error("0 queries parsed. Format: AILA_Q1||facts text"); return
    st.success(f"{n_q} queries loaded.")
    if not GROUND_TRUTH_RAW:
        st.error(f"Relevance judgments not found at {QRELS_PATH} (set AILA_QRELS)."); return

    st.markdown("### Step 2 — Pre-generated issues JSON")
    issues_path = st.text_input("Path to issues JSON",
        value=os.environ.get("AILA_ISSUES", str(REPO_ROOT / "data" / "fire_aila" / "issues.json")))
    if not issues_path.strip() or not os.path.isfile(issues_path.strip()):
        st.error("Issues JSON not found."); return
    issues_data = load_issues(issues_path.strip())
    uncovered = [q for q in query_texts if q not in issues_data]
    st.success(f"{len(issues_data)} queries have pre-generated issues.")
    if uncovered:
        st.warning(f"No issues for: {', '.join(sorted(uncovered))} — issue variants will return empty rankings for these.")

    st.markdown("### Step 3 — Document folder")
    folder = st.text_input("Path to GT_pool folder (TXT or JSON)",
        value=os.environ.get("AILA_DOCS", str(REPO_ROOT / "data" / "fire_aila" / "GT_pool")))
    if not folder.strip() or not os.path.isdir(folder.strip()):
        st.error("Folder not found."); return

    with st.spinner("Loading corpus…"):
        corpus = load_corpus(folder.strip())
    if not corpus:
        st.error("No documents loaded."); return
    st.success(f"{len(corpus)} documents loaded.")

    doc_texts = tuple(d["text"] for d in corpus)
    doc_ids   = [d["case_id"] for d in corpus]
    all_ids   = set(doc_ids)
    corpus_key = f"{folder.strip()}:{len(corpus)}"

    st.markdown("### Step 4 — Embedding model(s) × encoding strategy")
    st.caption("Chunk-pool encoding can be applied to ANY model, not just ones fine-tuned "
               "that way -- this separates 'does the encoding method matter' from 'does the "
               "fine-tuning matter'. Legacy mode truncates at max_seq_length like before. "
               "Start small (1 model × 1-2 encodings) on your first run to confirm things work "
               "before selecting everything at once.")

    selected_models = st.multiselect(
        "Base embedding model(s)", list(EMBED_MODELS.keys()),
        default=[list(EMBED_MODELS.keys())[-1]],  # default to just the fine-tuned model, not all 4
    )
    selected_modes = st.multiselect(
        "Encoding strategy", list(ENCODING_MODES.keys()),
        default=list(ENCODING_MODES.keys()),
    )

    if not selected_models or not selected_modes:
        st.error("Select at least one model and one encoding strategy.")
        return

    models_to_run = [
        (f"{m_label} — {enc_label}", EMBED_MODELS[m_label], enc_label)
        for m_label in selected_models
        for enc_label in selected_modes
    ]

    st.info(f"{len(models_to_run)} model × encoding combination(s) selected "
            f"({len(selected_models)} model(s) × {len(selected_modes)} encoding mode(s)) "
            f"× 6 variants × {len(WEIGHT_GRID)} weights × {n_q} queries — raw scores computed "
            f"once per (query, variant, combo), weight combination is cheap reuse. "
            f"No LLM calls at runtime.")

    if st.button("Run Issue Contribution × Weight Grid Analysis", type="primary"):
        with st.spinner("Building BM25 index…"):
            bm25_corpus = [preprocess(t) for t in doc_texts]
            bm25 = BM25Okapi(bm25_corpus)

        all_pq = {}
        progress = st.progress(0)
        for mi, (combined_label, model_name, encoding_mode_label) in enumerate(models_to_run):
            progress.progress(mi/len(models_to_run), text=f"Encoding corpus with {combined_label}…")
            with st.spinner(f"Encoding {len(doc_texts)} documents with {combined_label}…"):
                doc_embeddings = build_doc_embeddings(model_name, encoding_mode_label, corpus_key, doc_texts)
            progress.progress((mi+0.5)/len(models_to_run), text=f"Running grid retrieval for {combined_label}…")
            all_pq[combined_label] = run_embed_model_grid(
                model_name, encoding_mode_label, doc_texts, doc_ids, all_ids, bm25, doc_embeddings,
                query_texts, issues_data,
            )
            # Free the (potentially large) doc embedding matrix for this combo before
            # moving on -- get_embedder() itself stays cached (that's intentional, so
            # a repeated model across encoding modes doesn't reload from disk), but
            # doc_embeddings is per-combo and safe to drop here.
            del doc_embeddings
            gc.collect()
        progress.progress(1.0, text="Done!")

        st.session_state["ic_all_pq"]    = all_pq
        st.session_state["ic_n_docs"]    = len(corpus)
        st.session_state["ic_n_queries"] = n_q

    if st.session_state.get("ic_all_pq") is not None:
        if st.button("Clear results", type="secondary"):
            st.session_state["ic_all_pq"] = None
            st.rerun()
        display_results(
            st.session_state["ic_all_pq"],
            st.session_state["ic_n_docs"],
            st.session_state["ic_n_queries"],
        )


if __name__ == "__main__":
    main()