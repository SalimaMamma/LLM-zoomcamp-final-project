-- Papiers scientifiques bruts récupérés via Semantic Scholar
CREATE TABLE IF NOT EXISTS papers (
    paper_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    abstract        TEXT,
    year            INTEGER,
    venue           TEXT,
    study_type      TEXT,              -- meta-analysis / rct / observational / unknown
    citation_count  INTEGER DEFAULT 0,
    url             TEXT,
    query_used      TEXT,              -- requête d'ingestion ayant remonté ce papier
    fetched_at      TIMESTAMP DEFAULT now()
);

-- Chunks issus du découpage par section des papiers
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    paper_id        TEXT REFERENCES papers(paper_id),
    section         TEXT,              -- abstract / result / conclusion ...
    content         TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT now()
);

-- Entités et relations extraites pour le GraphRAG léger
CREATE TABLE IF NOT EXISTS graph_edges (
    id              SERIAL PRIMARY KEY,
    source_paper_id TEXT REFERENCES papers(paper_id),
    subject         TEXT NOT NULL,     -- ex: "intermittent fasting"
    relation        TEXT NOT NULL,     -- ex: "improves" / "contradicts" / "no_effect_on"
    object          TEXT NOT NULL,     -- ex: "endurance performance"
    created_at      TIMESTAMP DEFAULT now()
);

-- Feedback utilisateur collecté depuis l'interface Streamlit
CREATE TABLE IF NOT EXISTS feedback (
    id              SERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    rating          SMALLINT,          -- 1 = positif, -1 = négatif
    retrieval_mode  TEXT,              -- hybrid / graph
    latency_ms      INTEGER,
    created_at      TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_subject ON graph_edges(subject);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
