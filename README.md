# sbert-legal-retrieval

Code for the JURIX 2026 paper:

> **Why Semantic Search Struggles with Legal Text: A Diagnostic Study of Indian Case-Law Retrieval**
> Vishnuprabha V, Daleesha M Viswanathan, Rajesh R, Aarshitha Unnikrishnan.
> *Legal Knowledge and Information Systems (JURIX 2026)*.

The repository contains the pipeline that builds automatic supervision from Indian
Supreme Court judgments, fine-tunes a sentence-embedding bi-encoder
(`all-MiniLM-L6-v2`) for paragraph retrieval, and evaluates it, including the
differentiable **chunk-and-pool** encoding that removes the 256-token truncation
bottleneck at training and inference time.

## Overview

```
raw judgments + LLM summaries
        │
Stage 0  merge same-role paragraphs into segments (≤300 words, sentence-aware split)
Stage 1  align summary sentences to segments (cosine + BM25, RRF fusion, role bonus)
Stage 2  generate 3 queries per aligned pair with Llama-3.3-70B-Instruct (watsonx.ai)
Stage 3  build training table: every query gets two positives
         (original judgment paragraph + LLM summary sentence)
Stage 4  split by judgment into train / val / fixed held-out test (563 / 70 / 60)
        │
Stage 5  fine-tune with MultipleNegativesRankingLoss
         (source-mixed batches; legacy truncation or chunk-and-pool)
Stage 6  held-out evaluation: retrieval by role / query style + style-parity diagnostic
```

## Repository layout

| Path | Contents |
|---|---|
| `src/stage0_…` – `src/stage6_…` | Pipeline stages (Sections 3.2–3.4 and 5.3 of the paper) |
| `src/common/` | Shared code: chunk-and-pool encoding, BM25 / RRF, embeddings, Llama client, I/O |
| `run_pipeline.py` | Runs stages 0–4 in order |
| `ablation_encoding.py` | Legacy vs. chunk-pool-mean vs. chunk-pool-max, trained on identical data and order (Section 3.4) |
| `ablation_negatives.py` | In-batch vs. mined hard negatives (Section 3.3) |
| `learning_curve.py`, `evaluate_all_sizes.py` | Training-data-size analysis against the fixed holdout |
| `embed_long_text.py` | Embed an arbitrarily long text with chunk-and-pool at inference time |
| `explore_test_set.py`, `plot_training_curves.py`, `time_a_few_steps.py` | Inspection and diagnostics |
| `find_*.py` | Data-quality maintenance for stage outputs (oversized, empty or stale files) |
| `tools/review_app.py` | Streamlit dashboard for browsing every stage's output |
| `config.yaml` | All paths, thresholds and hyperparameters |
| `data/stage3_training_table/` | The training table and train / validation / test splits used in the paper (Parquet) |
| `data/fixed_holdout_judgment_ids.txt` | The 60 held-out test judgments used for every reported result |
| `data/samples/stage2_queries_sample.json` | Example Stage 2 output for one judgment (data format reference) |
| `evaluation/controlled_eval_app.py` | Section 5.1: BM25 vs. five models × two encodings on the controlled query set (Streamlit; batch and single-query modes) |
| `evaluation/summarize_controlled_eval.py` | Rebuilds Table 1 (with Wilson intervals and sign tests) and the per-query-type breakdown from the per-query results |
| `evaluation/multi_judgment_app.py` | Interactive hybrid retrieval across several uploaded judgments |
| `data/controlled_eval/controlled_evaluation_query_set.csv` | The 120 expert-validated queries (C14, C54) with gold paragraphs, roles, query style and overlap labels |
| `results/controlled_eval/` | Section 5.1 outputs: per-query results for all ten configurations, Table 1, Table 2 leaderboard, per-query-type table |

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vishnuprabha72/sbert-legal-retrieval.git
cd sbert-legal-retrieval
uv sync                       # CPU build of torch 2.6.0 (see pyproject.toml for GPU)
cp .env.example .env          # only needed for Stage 2; add your watsonx.ai credentials
```

Stage 2 calls Llama-3.3-70B-Instruct through [IBM watsonx.ai](https://www.ibm.com/watsonx). It needs a
watsonx.ai API key, project ID and region URL in `.env` (see `.env.example`); the model
id is set by `query_generation.llama_model` in `config.yaml`.

## Data

**Input format.** Put two files per judgment in `data/raw/`:

- `{id}.json`: a list of paragraphs (or `{"paragraphs": [...]}`), each with its text and a
  rhetorical-role label in the field `rhetorical_role_llm_annotated`
  (`FACT, ISSUE, ARG_P, ARG_R, REASON, HOLDING, ORDER, LAW, NONE`);
- `{id}_corrected_summary.json`: the structured case summary (Facts, Issues, Arguments,
  Reasoning, Final Judgment, Statutes and Precedents Referred). `summary_field_specs` in
  `config.yaml` maps summary fields to roles.

The source judgments are public Indian Supreme Court decisions, available from the
Legal Information Institute of India. They are not redistributed here.

**Released dataset.** The training table and the train / validation / test splits
used in the paper, including the LLM-generated summaries and queries, are in
`data/stage3_training_table/`. With these files in place, stages 0–4 can be skipped
and training and evaluation run directly. The queries and summaries were generated
with Llama 3.3 (Built with Llama).

**FIRE/AILA.** The FIRE 2019 AILA collection, queries and relevance judgments must be
obtained from the track organisers; they are not included.

## Running

```bash
# build the dataset (stages 0-4); try a dry run on 3 judgments first
uv run python run_pipeline.py --limit 3
uv run python run_pipeline.py

# train (chunk-and-pool by default; set training.chunk_pool.enabled: false for legacy)
uv run python src/stage5_train_sbert.py

# held-out evaluation -> data/stage3_training_table/eval_report_<model>.json
uv run python src/stage6_evaluate.py --model-path data/models/sbert-legal-v1

# ablations reported in the paper
uv run python ablation_encoding.py
uv run python ablation_negatives.py
```

Stages 0–3 can also be run on their own with `--limit N`. Stages 0–2 skip judgments
whose output already exists; pass `--force` to regenerate them.

### Controlled paragraph-retrieval evaluation (Section 5.1)

```bash
uv sync --extra eval
uv run streamlit run evaluation/controlled_eval_app.py
```

Choose **Batch CSV evaluation**, upload
`data/controlled_eval/controlled_evaluation_query_set.csv` and the two judgments as
`C14.txt` / `C54.txt` (or PDF). C14 is *Central Inland Water Transport Corporation v.
Brojo Nath Ganguly* (1986) and C54 is *Alister Anthony Pareira v. State of Maharashtra*
(2012); both are public Supreme Court judgments. Gold paragraph numbers refer to the
judgments' own paragraph numbering. Download the per-query CSV, then:

```bash
uv run python evaluation/summarize_controlled_eval.py --input <downloaded per-query csv>
```

The fine-tuned models are read from `data/models/sbert-legal-v1-legacy` and
`data/models/sbert-legal-v1-chunkpoolmean`. To use the released checkpoints instead:

```bash
export LEGACY_MODEL_PATH=vishnuprabha72/sbert-legal-v1-legacy
export CHUNKPOOL_MODEL_PATH=vishnuprabha72/sbert-legal-v1-chunkpoolmean
```

BM25 uses `rank_bm25.BM25Okapi` defaults on lowercased, stopword-filtered NLTK tokens.
Table 1 compares BM25 with Legacy-trained (legacy encoding); pass `--model` /
`--encoding` to `summarize_controlled_eval.py` for any other configuration.

Notes on reproducibility:

- Stage 2 calls an LLM, so regenerated queries will differ from the released splits.
  Use the files in `data/stage3_training_table/` to reproduce the reported numbers.
- Training runs are unseeded and each reported number comes from a single run.
- Do not regenerate `data/fixed_holdout_judgment_ids.txt`. All results are reported on
  this exact set of judgments.

## Models

The fine-tuned checkpoints are on the Hugging Face Hub:

- Legacy-trained: [`vishnuprabha72/sbert-legal-v1-legacy`](https://huggingface.co/vishnuprabha72/sbert-legal-v1-legacy)
- CP-trained: [`vishnuprabha72/sbert-legal-v1-chunkpoolmean`](https://huggingface.co/vishnuprabha72/sbert-legal-v1-chunkpoolmean)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("vishnuprabha72/sbert-legal-v1-legacy")
```

For texts longer than 256 tokens, use chunk-and-pool rather than `model.encode()`,
which truncates silently:

```bash
uv run python embed_long_text.py --model-path <model> --text-file judgment.txt
```

## Acknowledgements

We thank IBM for access to Llama-3.3-70B-Instruct through IBM watsonx.ai, which was
used to generate the training queries.

## Citation

```bibtex
@inproceedings{vishnuprabha2026semantic,
  title     = {Why Semantic Search Struggles with Legal Text: A Diagnostic Study of Indian Case-Law Retrieval},
  author    = {V, Vishnuprabha and Viswanathan, Daleesha M and R, Rajesh and Unnikrishnan, Aarshitha},
  booktitle = {Legal Knowledge and Information Systems (JURIX 2026)},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`). The license covers the code only. It does not cover the
source judgments, the FIRE/AILA data, or text generated by Llama models, which remain
under their own terms.

This is a research prototype and does not provide legal advice.
