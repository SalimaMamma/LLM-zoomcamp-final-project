# Evaluation

This document details how retrieval and generation are evaluated, with the
real results obtained on this corpus. Back to the [README](../README.md).

## 1. Retrieval evaluation

Two evaluations exist. **1b is the one to trust** — larger, and built on
exact gold labels instead of a keyword proxy. 1a is kept for the record
because it's what the re-ranking default was *originally* decided from,
and that decision got reversed once 1b existed — the point of documenting
both is showing the reversal happened because of better evidence, not
quietly editing history.

### 1a. Hand-annotated evaluation (original, 8 questions)

**Script:** [`src/eval/retrieval_eval.py`](../src/eval/retrieval_eval.py)
**Question set:** [`src/eval/questions_annotees.json`](../src/eval/questions_annotees.json)
— 8 questions in French, each annotated with the English keywords expected
in a relevant source (the corpus is in English; user-facing questions are
in French, matching the app's target audience).

#### Methodology

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

#### Why `hybrid` is the app's default *mode* despite these numbers

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

#### Re-ranking on this small set: looked harmful — superseded below

A cross-encoder re-ranking pass (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
see [`src/retrieval/hybrid.py`](../src/retrieval/hybrid.py)) fetches
`top_k × 3` candidates and re-scores each `(query, chunk)` pair. On this
8-question set:

| Mode | Recall@8 (off → on) | MRR (off → on) |
|---|:---:|:---:|
| vector | 1.00 → 1.00 | 0.938 → 0.906 |
| bm25 | 0.875 → 0.875 | 0.625 → 0.875 |
| hybrid | 1.00 → 0.875 | 0.917 → 0.875 |
| graph | 1.00 → 1.00 | 1.00 → 1.00 |

Read at face value, this said re-ranking helps `bm25` but hurts `hybrid`
— and the app's re-ranking checkbox defaulted to **off** on that basis.
**Section 1b below (24 questions, exact gold labels) shows the opposite
for `hybrid`**, and is the more trustworthy result: 8 questions is a small
sample, and — as section 1b's corpus-noise finding makes clear — this
kind of small hand-picked set can't surface some of what a larger,
automatically-sampled set catches. The current default (re-ranking **on**)
follows 1b, not this table. Kept here rather than deleted, because
reversing a decision silently once better evidence shows up is worse than
showing the reversal happened.

### 1b. Synthetic gold-label evaluation (24 questions — the one to trust)

**Scripts:**
[`src/eval/generate_synthetic_questions.py`](../src/eval/generate_synthetic_questions.py) (builds the question set) and
[`src/eval/retrieval_eval_synthetic.py`](../src/eval/retrieval_eval_synthetic.py) (runs the evaluation).
**Question set:** [`src/eval/synthetic_questions.json`](../src/eval/synthetic_questions.json).

#### Methodology: no manual annotation, and an exact gold label instead of a keyword proxy

1a's keyword-match proxy only confirms a chunk *contains a word* — it
never confirms the chunk is the one the answer actually came from. Here,
a Groq call reads a real indexed chunk (`result`/`conclusion` sections
only — they contain concrete findings, unlike `method`/`background`) and
writes 2 questions whose answer is stated in *that exact chunk*, plus a
short answer. The chunk's `chunk_id` and `paper_id` become **exact gold
labels** — no annotation effort, and no ambiguity about what "relevant"
means:

- **Recall@8**: did the gold `chunk_id` (or, more leniently, any chunk
  from the gold `paper_id`) appear in the top 8 results?
- **MRR**: reciprocal rank of that exact match.

18 papers were sampled (1 chunk each, so no paper dominates the set), 2
questions generated per chunk = 36 raw pairs.

#### A real finding this surfaced: ~33% of the sampled papers were off-topic

Before running any retrieval evaluation, reading the 36 generated
questions surfaced something the hand-picked 8-question set never could:
**6 of the 18 sampled papers are not about sports nutrition or athletic
performance at all** — e.g. *"Bone Morphogenetic Protein (BMP) signaling
in development and human diseases"*, *"...high-fat diet-induced MASLD..."*
(a liver disease), *"SGLT2 inhibition with empagliflozin..."* (a diabetes
drug in mice). Tracing them back to `query_used`
([src/ingestion/europepmc.py](../src/ingestion/europepmc.py)) shows
exactly how they got in: the query `"fasted cardio fat oxidation"`
matched a Framingham Heart Study paper about *pericardial fat* and
cardiovascular disease — Europe PMC's search matched the words "fat" and
"cardio(vascular)" with no notion that "cardio" means exercise here, not
cardiology.

**This is a corpus quality issue, not a retrieval or eval bug** — the
retriever is doing its job correctly on a corpus that itself contains
off-topic noise from loose keyword-based ingestion queries. The 12
off-topic questions (from those 6 papers) were dropped from the
evaluation set (36 → 24) so the retrieval numbers below measure retrieval
quality, not the retriever's ability to distinguish real vs. noise
papers — a separate problem. It's tracked as a known limitation (see the
README) rather than silently fixed by re-ingesting, which is a larger,
separate piece of work (a topical relevance filter on ingestion, or an
LLM-based query-relevance check per result).

#### Results (24 on-topic questions, exact chunk/paper match)

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

#### What changes with this stronger evidence

1. **Re-ranking now clearly helps every mode**, including `hybrid`
   (chunk recall@8 0.83→0.92, MRR 0.74→0.90) — the opposite of what 1a
   showed. **The app's re-ranking checkbox now defaults to on.** With 3x
   the sample size and exact labels instead of a keyword proxy, this
   result is trusted over 1a's.
2. **`graph` mode is genuinely weak**, not just "hard to score fairly."
   1a's hit-based metric made it look perfect (1.00/1.00) — documented at
   the time as a likely artifact of the metric, not real quality. This
   evaluation confirms that suspicion directly: paper-level recall of
   0.21-0.29 means the entity-matching approach
   (`find_matching_nodes()` in [`src/graphrag/graph_store.py`](../src/graphrag/graph_store.py),
   substring match against node names) misses the right source most of
   the time. `graph` remains available as a mode — the knowledge-graph
   traversal is still a genuinely different retrieval path worth having
   for relation-style questions ("does X contradict Y?") — but this is
   real evidence it shouldn't be anyone's default.
3. **`hybrid+rerank` and `vector+rerank` tie exactly** (0.92/0.90 both).
   `hybrid` stays the app's default *mode* regardless — the robustness
   argument in 1a (BM25's lexical signal as a hedge against embeddings
   missing an exact term) is about mode choice, not re-ranking, and
   isn't contradicted by this result.

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
