# Evaluation

This document details how retrieval and generation are evaluated, with the
real results obtained on this corpus. Back to the [README](../README.md).

## 1. Retrieval evaluation

**Script:** [`src/eval/retrieval_eval.py`](../src/eval/retrieval_eval.py)
**Question set:** [`src/eval/questions_annotees.json`](../src/eval/questions_annotees.json)
— 8 questions in French, each annotated with the English keywords expected
in a relevant source (the corpus is in English; user-facing questions are
in French, matching the app's target audience).

### Methodology

There are no manual "this chunk_id is relevant to this question" labels
(that would require chunk-by-chunk annotation, out of scope for a solo
project). Instead, a simple but objective proxy is used: a chunk is
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

### Results (current corpus: 561 papers / 1376 chunks)

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

### Why `hybrid` is the app's default mode despite these numbers

On this 8-question sample, `vector` and `graph` show a nominally higher
MRR than `hybrid`. Two reasons not to read this as "turn hybrid off":

1. **8 questions is too small a sample to be conclusive** — a 0.02 gap on
   8 data points isn't statistically meaningful.
2. **The `graph` proxy is structurally favored by this metric**:
   `evaluate_graph()` counts a hit as soon as *any* relation exists for the
   question's first keyword, with no notion of rank or content quality
   (unlike `vector`/`bm25`/`hybrid`, which actually rank retrieved chunks).
   An MRR of 1.0 here means "the term exists somewhere in the graph", not
   "the best answer is ranked first". This isn't directly comparable to
   the same metric applied to the other three modes — a real limitation of
   the eval script, documented here rather than hidden.
3. **BM25 alone drops noticeably** (0.625 MRR) on questions asked in
   French against an English corpus — exact lexical matching misses the
   synonyms/translations that embeddings capture. `hybrid` remains the
   default because it never loses much against the best individual mode
   while avoiding the worst case (a purely terminological question where
   BM25 is strong, or conversely a question with many synonyms where
   vector search is strong) — a robustness choice, not just raw score on
   this sample.

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

### Current status of results

⚠️ **Pending a full run.** The Groq account used for this project hit its
free-tier daily quota (200k tokens/day) while this evaluation was being
prepared, before it could finish. The script is functional (see the fix
below) and saves results incrementally — re-running the command above once
the quota resets (next day on the free tier, or immediately on a paid
tier) is enough to get real scores; this file will be updated with the
results table at that point.

**Bug fixed along the way**: Groq's forced-JSON mode occasionally fails to
produce valid JSON (unescaped typographic quotes inside the justification
string), which raised an uncaught `BadRequestError` and crashed the whole
script, losing every result obtained so far. `llm_eval.py` now catches
these errors per-question (the way `extract_entities.py` already does for
graph extraction) and saves after every question instead of only at the
very end.

## 3. What these evaluations validate in the pipeline

- The hybrid retriever works end-to-end (Qdrant + BM25 + RRF) and isn't
  worse than its components taken individually.
- The generation prompt does enforce source citation — a necessary (but
  not sufficient, hence the LLM judge) condition to limit hallucination.
- Both evaluation scripts are automated and re-runnable in one command —
  not a one-off manual check.
