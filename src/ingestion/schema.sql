-- Papiers scientifiques bruts récupérés via Europe PMC / OpenAlex
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

-- Log de CHAQUE requête (pas seulement celles avec un 👍/👎 explicite) --
-- alimente les métriques d'observabilité : latence par étape, coût,
-- erreurs, fraîcheur du corpus, qualité du retrieval. Voir docs/monitoring.md.
CREATE TABLE IF NOT EXISTS retrieval_log (
    id                 SERIAL PRIMARY KEY,
    question           TEXT NOT NULL,
    retrieval_mode     TEXT,              -- hybrid / vector / bm25 / graph
    embedding_ms       INTEGER,           -- temps d'encodage de la requête (0 si non applicable au mode)
    search_ms          INTEGER,           -- temps de recherche (Qdrant+BM25+RRF, ou traversée du graphe)
    generation_ms       INTEGER,           -- temps d'appel Groq pour générer la réponse
    prompt_tokens      INTEGER,           -- tokens en entrée du prompt de génération
    completion_tokens  INTEGER,           -- tokens en sortie
    avg_chunk_score    REAL,              -- score moyen des chunks retournés (NULL en mode graph, pas de score)
    min_chunk_score    REAL,
    chunk_count        INTEGER,           -- nombre de chunks retournés (0 = aucun résultat)
    paper_ids          TEXT[],            -- paper_id distincts retournés, pour la diversité des sources
    answer_length      INTEGER,           -- longueur de la réponse en caractères
    is_declined        BOOLEAN,           -- réponse du type "aucune source pertinente" (comportement voulu, pas un échec)
    is_well_formatted  BOOLEAN,           -- citation + ligne de niveau de preuve présentes
    is_error           BOOLEAN DEFAULT false,
    error_message      TEXT,
    created_at         TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_subject ON graph_edges(subject);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_log_created_at ON retrieval_log(created_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_log_mode ON retrieval_log(retrieval_mode);
