# Monitoring

Back to the [README](../README.md).

## How feedback is collected

Every answer shown in the Streamlit app ([streamlit_app/app.py](../streamlit_app/app.py))
has two buttons, **👍 Utile** ("Useful") / **👎 Pas utile** ("Not useful").
A click inserts a row into the Postgres `feedback` table:

```sql
INSERT INTO feedback (question, answer, rating, retrieval_mode, latency_ms)
VALUES (%s, %s, %s, %s, %s)
```

`rating` (+1/-1), `retrieval_mode` (hybrid/vector/bm25/graph), and
`latency_ms` (measured client-side, from clicking "Vérifier" to the
answer being generated) are recorded every time — not just an aggregate
counter — which makes it possible to break down by mode or over time.

## Grafana dashboard

Auto-provisioned at startup (`monitoring/provisioning/`, no manual Grafana
UI configuration needed) — available as soon as `docker compose up -d`
runs, at http://localhost:3001 (`admin`/`admin`), folder **SciFit-Check**.

![Grafana dashboard](images/03_grafana_dashboard.png)

6 panels ([source JSON](../monitoring/provisioning/dashboards/scifit_overview.json)):

| Panel | Type | Source | What it shows |
|---|---|---|---|
| Volume de questions par jour (questions/day) | timeseries | `feedback` | Usage over time |
| Taux de satisfaction (%) (satisfaction rate) | stat | `feedback` | % of positive ratings |
| Latence moyenne (ms) (average latency) | stat | `feedback` | End-to-end average response time |
| Répartition des modes de retrieval utilisés (retrieval mode split) | piechart | `feedback` | hybrid vs vector vs bm25 vs graph |
| Feedback positif vs négatif par mode (positive vs negative feedback by mode) | barchart | `feedback` | Perceived quality per retrieval mode |
| Papiers indexés par niveau de preuve (papers by evidence level) | piechart | `papers` | Corpus composition (meta-analysis/rct/observational/unknown) |

The last panel doesn't depend on the `feedback` table — it reflects the
state of the indexed corpus and works right after ingestion, before any
user interaction. It's the one visible in the screenshot above: **462 out
of 561 papers (82%) are classified `unknown`** — a real blind spot worth
noting honestly rather than hiding: `classify_study_type()`
([src/ingestion/europepmc.py](../src/ingestion/europepmc.py)) relies on
`pubTypeList`, available and reliable from Europe PMC but absent from
OpenAlex metadata, which feeds a large share of the corpus. Improvement
path: classify `study_type` heuristically from title/abstract for
OpenAlex-sourced papers, or only use OpenAlex as a targeted complement.

### The other 5 panels (based on `feedback`)

As of writing, the `feedback` table is empty (0 interactions) and these
panels show "No data" — not a bug, just no real usage yet. Two ways to
populate them:

1. **Normal usage**: open the app, ask questions, click 👍/👎.
2. **Demo seed script** ([`src/eval/seed_feedback_demo.py`](../src/eval/seed_feedback_demo.py)):
   runs the real pipeline (retrieval + Groq generation) over the 8
   annotated questions × 4 modes, and logs each interaction into
   `feedback` exactly like the app would. The rating isn't a randomly
   simulated human opinion: it's derived from an automatic check of the
   expected answer format (source citation + "evidence level" line
   present, per the prompt in `src/llm/answer.py`).
   ```bash
   docker compose exec app python src/eval/seed_feedback_demo.py
   ```
   ⚠️ Uses ~32 Groq calls — see
   [Groq quota](setup.md#groq-free-tier-quota) if the account is already
   close to its daily limit when running this.

A configuration bug was fixed on both `piechart` panels while preparing
this documentation: without an explicit `reduceOptions.values: true`,
Grafana (v11) collapses every row of a table query into a single
aggregated value named "total" instead of drawing one slice per category —
caught by actually testing the dashboard with real data rather than
committing the config untested.

## What's missing to go further

- **Alerting**: Grafana supports it natively (the "Alert rules" tab is
  already visible in the provisioned UI) — not configured here, but a
  threshold on satisfaction rate or latency would be the logical next
  step.
- **Structured application logs**: currently only explicit feedback is
  tracked; errors (e.g. empty retrieval, a failed Groq call) only surface
  as `st.warning()` in the UI, not in a queryable table.
