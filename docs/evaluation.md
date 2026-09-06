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
   the eval script, documented here rather than hidden. (A related bug was
   fixed alongside this: `evaluate_graph()` used to pull from an
   *uncapped* pool of relations — a broadly-matching term can return 100+
   — while every other mode was capped at `top_k`, an unfair comparison.
   Now capped the same way; numbers were unchanged on this 8-question set,
   but the comparison is honest rather than accidentally lopsided. See
   [docs/monitoring.md](monitoring.md) for the production-side version of
   the same bug, caught with real token-cost numbers.)
3. **BM25 alone drops noticeably** (0.625 MRR) on questions asked in
   French against an English corpus — exact lexical matching misses the
   synonyms/translations that embeddings capture. `hybrid` remains the
   default because it never loses much against the best individual mode
   while avoiding the worst case (a purely terminological question where
   BM25 is strong, or conversely a question with many synonyms where
   vector search is strong) — a robustness choice, not just raw score on
   this sample.

### Re-ranking: measured, not assumed

A cross-encoder re-ranking pass (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
see [`src/retrieval/hybrid.py`](../src/retrieval/hybrid.py)) is available
in every mode: fetch `top_k × 3` candidates, score each `(query, chunk)`
pair with the cross-encoder, keep the top `k`. It's implemented and
**evaluated**, exactly like hybrid search itself, rather than assumed to
help just because it's a known technique:

| Mode | Recall@8 (off → on) | MRR (off → on) |
|---|:---:|:---:|
| vector | 1.00 → 1.00 | 0.938 → 0.906 |
| bm25 | 0.875 → 0.875 | **0.625 → 0.875** |
| **hybrid** *(app default)* | **1.00 → 0.875** | 0.917 → 0.875 |
| graph | 1.00 → 1.00 | 1.00 → 1.00 |

**Re-ranking is a clear win for `bm25`** (MRR +0.25) — makes sense: raw
BM25 ranking is purely lexical and the cross-encoder adds real semantic
judgment on top. **It measurably hurts `hybrid`**, the app's default mode:
recall@8 drops from 1.00 to 0.875, meaning on this 8-question set, at
least one previously-correct answer got pushed out of the top 8 by the
re-ranking pass. RRF-fused rankings are already a decent blend of
lexical + semantic signal; re-ranking on top of that can apparently
overrule a correct RRF ranking with a cross-encoder judgment that,
on a small sample, is sometimes wrong.

**Decision made from this data, not despite it**: the app's re-ranking
checkbox defaults to **off**. It's still available as a toggle (and used
in the monitoring demo seed, to keep the `rerank_ms`/`reranked` telemetry
populated) but the measured evidence for the default mode doesn't support
turning it on by default. As with the hybrid-vs-vector comparison above,
8 questions is a small sample — this conclusion could change with a
larger annotated set, and re-running
`docker compose exec app python src/eval/retrieval_eval.py` is one
command away if the corpus or question set grows.

Reproduce with the same command as the retrieval evaluation above; the
8 `+rerank` rows are included automatically in
[`data/eval_results/retrieval_eval.json`](../data/eval_results/retrieval_eval.json).

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

Run across two sessions two days apart (the Groq free-tier daily quota ran
out mid-way through session 1; the resume-aware save — see the fix below —
carried the first 4 results over instead of losing them, and session 2
picked up exactly where it left off):

| Strategy | n scored | Avg. faithfulness |
|---|:---:|:---:|
| **`zero_shot`** | 7/8 | **9.7 / 10** |
| `structured_evidence` | 7/8 | 9.3 / 10 |

(1 question per strategy has `faithfulness_score: null` — the judge's
forced-JSON call failed for those two, see the bug note below; excluded
from the average rather than guessed at.)

Full answers and per-question justifications:
[`data/eval_results/llm_eval.json`](../data/eval_results/llm_eval.json).

`zero_shot` edges out `structured_evidence` slightly. The one clearly
lower score in each strategy came from the same underlying issue: on the
protein-timing question, the model asserted a general claim ("a 30-min
window isn't strictly required") not directly stated in the retrieved
excerpts, even though every per-source citation it gave was accurate —
`structured_evidence`'s more elaborate 3-part format apparently invites
slightly more of this over-generalizing-from-accurate-citations pattern
than the plainer `zero_shot` prompt. This lines up with the production
prompt in [`src/llm/answer.py`](../src/llm/answer.py), which follows the
same free-form-with-citations pattern as `zero_shot` rather than the more
structured format — the evaluation supports the prompt actually shipped,
not just picks a winner after the fact.



## 3. What these evaluations validate in the pipeline

- The hybrid retriever works end-to-end (Qdrant + BM25 + RRF) and isn't
  worse than its components taken individually.
- The generation prompt does enforce source citation — a necessary (but
  not sufficient, hence the LLM judge) condition to limit hallucination.
- Both evaluation scripts are automated and re-runnable in one command —
  not a one-off manual check.
