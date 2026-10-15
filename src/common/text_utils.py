"""
Sentence-aware text chunking, shared by:
  - stage0_merge_segments.py: splitting an individual raw paragraph that's
    ALREADY over max_segment_words on its own (a data-quality issue where
    upstream paragraph-splitting failed and lumped a large block of text --
    e.g. an entire annexure/schedule -- into one paragraph object). The
    per-merge-attempt word-count check in merge_segments() only fires when
    ATTEMPTING TO COMBINE two paragraphs; it never validates a single
    input that's already oversized, so such a paragraph would otherwise
    pass through untouched (the largest we found was ~34,000 words).
  - Inference-time embedding of long real-world paragraphs (see
    chunk_and_pool_embed below): the trained model's max_seq_length (256
    tokens) truncates anything longer silently -- for a ~1000-word
    paragraph, only the first ~150-200 words would ever reach the model
    without chunking, with the remainder invisible to the resulting
    embedding.

Chunking is done at SENTENCE boundaries (greedy bin-packing sentences up
to max_words per chunk), not mid-sentence, to avoid producing
fragments that end abruptly.
"""
import re

import numpy as np

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def chunk_text_by_words(text: str, max_words: int) -> list[str]:
    """
    Greedily packs sentences into chunks of at most `max_words` words each.

    A single "sentence" longer than max_words is common in this corpus
    (semicolon-separated clauses, numbered sub-clauses, long unbroken
    passages that SENT_SPLIT_RE cannot split). Such a sentence is split at
    word boundaries as a last resort, as chunk_text_by_tokens does for
    tokens.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text] if len(text.split()) else []

    chunks = []
    current: list[str] = []
    current_wc = 0
    for sentence in sentences:
        words = sentence.split()
        wc = len(words)

        if wc > max_words:
            # Oversized single sentence -- flush pending, then hard-split
            # this sentence's own words into <= max_words pieces.
            if current:
                chunks.append(" ".join(current))
                current = []
                current_wc = 0
            for i in range(0, wc, max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue

        if current and current_wc + wc > max_words:
            chunks.append(" ".join(current))
            current = []
            current_wc = 0
        current.append(sentence)
        current_wc += wc
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_and_pool_embed(text: str, model, max_words: int = 250, pooling: str = "mean") -> np.ndarray:
    """
    Embeds a long paragraph by splitting it into chunks (via
    chunk_text_by_words), embedding each chunk separately, and pooling the
    results into one vector -- so a long real-world paragraph's full
    content contributes to the final embedding instead of only whatever
    fits in the model's first max_seq_length tokens.

    max_words=250 is a conservative word-level proxy for staying under
    the model's 256-TOKEN limit per chunk (legal text tokenizes to more
    subword pieces per word than plain English, so 250 words is not a
    hard guarantee -- see chunk_and_pool_embed_by_tokens for an exact,
    tokenizer-aware version if you need precision).

    pooling: "mean" (default -- represents the paragraph's overall content)
             or "max" (element-wise max across chunks -- can better surface
             a single standout chunk's signal, at the cost of the pooled
             vector no longer corresponding to any one real chunk).
    """
    chunks = chunk_text_by_words(text, max_words)
    if not chunks:
        raise ValueError("chunk_and_pool_embed: input text is empty after cleaning")

    chunk_embs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    chunk_embs = np.atleast_2d(chunk_embs)

    if pooling == "mean":
        pooled = chunk_embs.mean(axis=0)
    elif pooling == "max":
        pooled = chunk_embs.max(axis=0)
    else:
        raise ValueError(f"Unknown pooling method: {pooling!r} (expected 'mean' or 'max')")

    # re-normalize after pooling -- mean/max of unit vectors isn't itself unit-length
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled


def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    """
    Sentence-aware chunking by ACTUAL subword token count (via the given
    tokenizer), not a word-count proxy. Extracted as its own function so
    training-time chunk-and-pool encoding (see
    src/common/chunk_pool_encoding.py) and this module's inference-time
    chunk_and_pool_embed_by_tokens split any given paragraph into IDENTICAL
    chunk boundaries -- otherwise a paragraph's train-time embedding and
    its eval/inference-time embedding could quietly diverge just because
    the two code paths chunked it slightly differently.

    Edge case: a single sentence that is itself longer than max_tokens
    (long statutory clauses, citation strings, or numbered sub-clauses
    without terminal punctuation often exceed 256 tokens). Never cutting
    mid-sentence is the right default, but letting an oversized sentence
    through untouched just reintroduces silent truncation the moment it
    reaches the model -- exactly the flaw this function exists to prevent,
    for this one case. So: if a single sentence alone exceeds max_tokens,
    it is hard-split at the TOKEN level (via the tokenizer's own encode/
    decode) into <= max_tokens pieces -- a mid-sentence cut only for this
    specific, unavoidable case, strictly better than the alternative of
    silent truncation.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()] or [text]

    chunks = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        n_tokens = len(token_ids)

        if n_tokens > max_tokens:
            # Oversized single sentence -- flush whatever's pending, then
            # hard-split this sentence's own tokens into <= max_tokens
            # pieces and add each as its own chunk.
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            for i in range(0, n_tokens, max_tokens):
                piece_ids = token_ids[i:i + max_tokens]
                chunks.append(tokenizer.decode(piece_ids, skip_special_tokens=True))
            continue

        if current and current_tokens + n_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += n_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_and_pool_embed_by_tokens(text: str, model, max_tokens: int = None, pooling: str = "mean") -> np.ndarray:
    """
    Exact, tokenizer-aware version of chunk_and_pool_embed -- chunks by
    actual subword token count (using the model's own tokenizer) rather
    than a word-count proxy, guaranteeing no chunk silently exceeds
    max_seq_length regardless of how legal-heavy the vocabulary is.
    """
    max_tokens = max_tokens or getattr(model, "max_seq_length", 256)

    text = (text or "").strip()
    if not text:
        raise ValueError("chunk_and_pool_embed_by_tokens: input text is empty")

    chunks = chunk_text_by_tokens(text, model.tokenizer, max_tokens)

    chunk_embs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    chunk_embs = np.atleast_2d(chunk_embs)

    if pooling == "mean":
        pooled = chunk_embs.mean(axis=0)
    elif pooling == "max":
        pooled = chunk_embs.max(axis=0)
    else:
        raise ValueError(f"Unknown pooling method: {pooling!r} (expected 'mean' or 'max')")

    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled
