# Monitoring

Back to the [README](../README.md).

## How feedback is collected

Every answer shown in the Streamlit app ([streamlit_app/app.py](../streamlit_app/app.py))
has two buttons, **👍 Useful** / **👎 Not useful**. A click inserts a row
into the Postgres `feedback` table (`rating`, `retrieval_mode`,
`latency_ms`).

**Every request is also logged**, not just the ones a user happens to
rate, into a separate `retrieval_log` table — this is what powers the
observability panels below (latency per stage, cost, retrieval quality,
corpus freshness, error rate). `feedback` answers "did the user like it?";
`retrieval_log` answers "what actually happened, technically, on every
single call?". See the schema in
[`src/ingestion/schema.sql`](../src/ingestion/schema.sql).

## Grafana dashboard

Auto-provisioned at startup (`monitoring/provisioning/`, no manual Grafana
UI configuration needed) — available as soon as `docker compose up -d`
runs, at http://localhost:3001 (`admin`/`admin`), folder **SciFit-Check**.

![Grafana dashboard](images/03_grafana_dashboard.png)

**17 panels** ([source JSON](../monitoring/provisioning/dashboards/scifit_overview.json)),
organized in four rows:

### Row 1-2 — usage & corpus (the original 6)

| Panel | Type | Source | What it shows |
|---|---|---|---|
| Questions per day | timeseries | `feedback` | Usage over time |
| Satisfaction rate (%) | stat | `feedback` | % of positive ratings |
| Average latency (ms) | stat | `feedback` | End-to-end average response time |
| Retrieval mode split | piechart | `feedback` | hybrid vs vector vs bm25 vs graph |
| Positive vs negative feedback by mode | barchart | `feedback` | Perceived quality per retrieval mode |
| Indexed papers by evidence level | piechart | `papers` | Corpus composition (meta-analysis/rct/observational/unknown) |

The last one doesn't depend on `feedback` — it reflects the indexed corpus
and works right after ingestion. It's the one showing **462 out of 561
papers (82%) classified `unknown`** — a real blind spot: `classify_study_type()`
([src/ingestion/europepmc.py](../src/ingestion/europepmc.py)) relies on
`pubTypeList`, reliable from Europe PMC but absent from OpenAlex metadata,
which feeds a large share of the corpus.

### Row 3 — performance & cost

| Panel | Type | Shows |
|---|---|---|
| Average latency by stage (ms) | barchart | embedding / search / generation, averaged separately |
| Error rate (%) | stat | % of requests where the pipeline raised an exception |
| Avg tokens per request (in/out) | barchart | Groq `prompt_tokens` / `completion_tokens`, direct proxy for $ cost per request |

**Real finding, not a hypothesis**: generation (~7.4s) dominates
end-to-end latency by roughly two orders of magnitude over embedding
(~50ms) and search (~25ms) combined — visible directly on the chart, where
the embedding/search bars are barely tall enough to show a value. If
latency ever needs optimizing, the LLM call is where the budget is, not
the retrieval side.

### Row 4 — generation quality & corpus freshness

| Panel | Type | Shows |
|---|---|---|
| Avg publication year of retrieved chunks | stat | How recent the evidence actually being surfaced is |
| Papers never retrieved (archival candidates) | stat | Indexed papers that have never come back in any query |
| "No relevant source" rate (%) | stat | How often the model correctly declines instead of hallucinating |
| Well-formatted answer rate (%) | stat | % with both a citation and an evidence-level line |

On the current (small, synthetic 32-request) seed: avg year **2016**,
**307/561 papers never retrieved**, decline rate **21.9%**, well-formatted
rate **53.1%**. That well-formatted number is a genuine, slightly
uncomfortable finding: even with an explicit prompt instruction, the model
skips the citation or the evidence-level line in roughly half of answers
in this sample — worth watching as real usage accumulates, not just
assumed to be near-100% because the prompt asks for it.

### Row 5 — retrieval quality

| Panel | Type | Shows |
|---|---|---|
| Avg chunk similarity score by mode | timeseries | One series per retrieval mode |
| Zero-result query rate (%) | stat | % of queries where retrieval returned literally nothing |
| Distinct papers retrieved (30d) | stat | Source diversity — are we always pulling the same handful of papers? |
| Top 10 most-retrieved papers | table | Which specific papers dominate |

**Why "zero-result rate" instead of a similarity threshold**: `vector`
mode returns a true cosine similarity (0-1), `bm25` returns an unbounded
lexical score, `hybrid` returns an RRF score (~0.01-0.03 range by
construction), and `graph` returns no score at all. These are not on a
comparable scale — the dashboard makes this visible rather than hiding
it: in the screenshot above, `bm25`'s score sits around 6 while
`hybrid`/`vector` sit near 0 on the same axis. A single hardcoded
similarity threshold across modes would silently flag one mode as
"always confident" and another as "always uncertain" regardless of actual
quality. So "no relevant chunk found" is defined as **zero chunks
returned** (mode-agnostic, unambiguous) instead. Per-mode calibrated
thresholds would be a reasonable follow-up, not implemented here.

## Populating the dashboard

Two ways:

1. **Normal usage**: open the app, ask questions, click 👍/👎. Every
   request (rated or not) also lands in `retrieval_log` automatically.
2. **Demo seed script** ([`src/eval/seed_feedback_demo.py`](../src/eval/seed_feedback_demo.py)):
   runs the real pipeline (retrieval + Groq generation) over the 8
   annotated questions × 4 modes, logging into both tables exactly like
   the app would. The rating isn't a randomly simulated human opinion:
   it reuses the same deterministic checks the app itself computes for
   monitoring (`src/llm/answer.py::is_declined_answer` /
   `is_well_formatted_answer`).
   ```bash
   docker compose exec app python src/eval/seed_feedback_demo.py
   ```
   ⚠️ Uses ~32 Groq calls, paced 8s apart to stay under Groq's 8,000
   tokens/minute limit — see [Groq quota](setup.md#groq-free-tier-quota).

## Bugs found by actually testing this dashboard, not just committing config

1. **Both `piechart` panels showed one merged slice instead of real
   categories.** Without an explicit `reduceOptions.values: true`, Grafana
   (v11) collapses every row of a table query into a single aggregated
   value named "total". Fixed in the panel `options`.
2. **All `feedback`-based panels showed "No data" despite the table
   having real rows.** The default time range was "Last 6 hours", but
   `date_trunc('day', created_at)` buckets every row inserted today down
   to *today at 00:00* — already outside a 6-hour window by the time the
   data was queried, and Grafana's panel clips data points outside the
   selected range client-side even though the SQL itself ignores the time
   picker. Fixed by setting the dashboard's default range to `now-30d` →
   `now` (top-level `"time"` key in the dashboard JSON).
3. **A regex was too strict and mis-rated genuinely good answers as
   negative.** The seed script originally required the literal
   `[source: n]` format from the prompt, but the model often cites as
   plain `[n]`. Fixed by accepting both, and by recognizing an explicit
   "no relevant source" decline as correct behavior rather than a
   formatting failure — both checks now live once in
   `src/llm/answer.py`, shared by the app and the seed script instead of
   duplicated.

None of these three were visible from reading the code or the JSON —
all three only surfaced once real data existed and the dashboard was
actually opened in a browser.

## What's missing to go further

- **Alerting**: Grafana supports it natively (the "Alert rules" tab is
  already visible in the provisioned UI) — not configured here, but a
  threshold on error rate or well-formatted rate would be the logical
  next step now that both are tracked.
- **Query-vocabulary drift** (are incoming questions drifting toward
  topics/terms the corpus doesn't cover well?) and **embedding-model
  drift** (relevant only if `EMBEDDING_MODEL` is ever changed) were
  considered and deliberately left out — see the README's
  [best-practices self-assessment](../README.md#known-limitations--self-assessment)
  for why.
