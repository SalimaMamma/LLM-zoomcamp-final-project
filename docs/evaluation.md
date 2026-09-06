# Evaluation

This document details how retrieval and generation are evaluated, with the
real results obtained on this corpus. Back to the [README](../README.md).

## 1. Retrieval evaluation

### 1a. Hand-annotated evaluation (original, 8 questions)

**Script:** [`src/eval/retrieval_eval.py`](../src/eval/retrieval_eval.py)
**Question set:** [`src/eval/questions_annotees.json`](../src/eval/questions_annotees.json)
— 8 questions in French, each annotated with the English keywords expected
in a relevant source (the corpus is in English; user-facing questions are
in French, matching the app's target audience).

#### Methodology

There are no manual "this chunk_id is relevant to this question" labels
(that would require chunk-by-chunk annotation). Instead, a simple but objective proxy is used: a chunk is
considered "relevant" if it contains at least one of the expected
keywords. For each retrieval mode, on each question:

- **Recall@8**: 1 if a relevant chunk appears in the top 8 results, 0
  otherwise.
- **MRR** (Mean Reciprocal Rank): `1 / rank` of the first relevant chunk
  (0 if none found).

Four modes are compared: `vector` (embeddings only), `bm25` (lexical
only), `hybrid` (vector+BM25 RRF fusion — see
[`src/retrieval/hybrid.py`](../src/retrieval/hybrid.py)), and `graph`
(traversal of the LLM-extracted knowledge graph).

#### Results (current corpus: 561 papers / 1376 chunks)

| Mode | Recall@8 | MRR |
|---|:---:|:---:|
| vector | 1.00 | 0.938 |
| bm25 | 0.875 | 0.625 |
| **hybrid** | **1.00** | **0.917** |
| graph | 1.00 | 1.00 |

Reproduce with:
```bash
docker compose exec app python src/eval/retrieval_eval.py
```
Raw results: [`data/eval_results/retrieval_eval.json`](../data/eval_results/retrieval_eval.json).
|

### 1b. Synthetic gold-label evaluation (generate questions — the one to trust more)

**Scripts:**
[`src/eval/generate_synthetic_questions.py`](../src/eval/generate_synthetic_questions.py) (builds the question set) and
[`src/eval/retrieval_eval_synthetic.py`](../src/eval/retrieval_eval_synthetic.py) (runs the evaluation).
**Question set:** [`src/eval/synthetic_questions.json`](../src/eval/synthetic_questions.json).

#### Methodology: no manual annotation, and an exact gold label
1a's keyword-match proxy only confirms a chunk *contains a word* — it
never confirms the chunk is the one the answer actually came from. Here,
a Groq call reads a real indexed chunk and
writes 2 questions whose answer is stated in *that exact chunk*, plus a
short answer. The chunk's `chunk_id` and `paper_id` become **exact gold
labels** 

- **Recall@8**: did the gold `chunk_id` (or, more leniently, any chunk
  from the gold `paper_id`) appear in the top 8 results?
- **MRR**: reciprocal rank of that exact match.

18 papers were sampled (1 chunk each, so no paper dominates the set), 2
questions generated per chunk = 36 raw pairs.

#### Results 

| Mode | Chunk Recall@8 | Chunk MRR | Paper Recall@8 | Paper MRR |
|---|:---:|:---:|:---:|:---:|
| vector | 0.83 | 0.52 | 0.83 | 0.53 |
| **vector+rerank** | **0.92** | **0.90** | 0.92 | 0.90 |
| bm25 | 0.75 | 0.68 | 0.75 | 0.70 |
| bm25+rerank | 0.79 | 0.76 | 0.79 | 0.76 |
| hybrid | 0.83 | 0.74 | 0.88 | 0.75 |
| **hybrid+rerank** *(app default)* | **0.92** | **0.90** | 0.92 | 0.90 |
| graph | 0.00\* | 0.00\* | 0.21 | 0.15 |
| graph+rerank | 0.00\* | 0.00\* | 0.29 | 0.26 |

*\*Chunk-level recall is always 0 for `graph`: graph relations aren't
indexed chunks, so there's no `chunk_id` to match — only the paper-level
columns are meaningful for this mode.*

Reproduce with:
```bash
docker compose exec app python src/eval/generate_synthetic_questions.py  # optional: regenerates the question set
docker compose exec app python src/eval/retrieval_eval_synthetic.py
```
Raw results: [`data/eval_results/retrieval_eval_synthetic.json`](../data/eval_results/retrieval_eval_synthetic.json).

## 2. LLM generation evaluation (LLM-as-judge)

**Script:** [`src/eval/llm_eval.py`](../src/eval/llm_eval.py)

Compares two prompting strategies for `generate_answer()`:

- `zero_shot`: free-form answer with `[source: n]` citations.
- `structured_evidence`: answer structured into 3 parts (one-sentence
  summary / cited evidence / limitations-and-contradictions).

For each question × strategy, a second LLM call ("judge", same Groq model,
temperature 0, forced JSON output) scores the **faithfulness** (0-10) of
the answer to the provided excerpts — i.e. whether every claim is
traceable to a source, with no hallucination.

Reproduce with:
```bash
docker compose exec app python src/eval/llm_eval.py
```
Raw results: `data/eval_results/llm_eval.json`.

### Results (complete — 8 of 8 questions)

| Strategy | n scored | Avg. faithfulness |
|---|:---:|:---:|
| **`zero_shot`** | 7/8 | **9.7 / 10** |
| `structured_evidence` | 7/8 | 9.3 / 10 |



## 3. What these evaluations validate in the pipeline

- The hybrid retriever works end-to-end (Qdrant + BM25 + RRF) and isn't
  worse than its components taken individually.
- The generation prompt does enforce source citation — a necessary (but
  not sufficient, hence the LLM judge) condition to limit hallucination.
- Both evaluation scripts are automated and re-runnable in one command —
  not a one-off manual check.
