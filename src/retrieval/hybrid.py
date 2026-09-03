"""
Retrieval hybride : fusionne les résultats vecteur (Qdrant) et lexical
(BM25) avec Reciprocal Rank Fusion (RRF), une méthode simple et robuste
pour combiner deux classements sans avoir à calibrer des poids.

Optionnellement, un reranking par cross-encoder (voir rerank_chunks) peut
être appliqué en second passage sur une short-list de candidats bruts,
avant troncature au top_k final envoyé au LLM -- voir docs/evaluation.md
pour la comparaison mesurée avec/sans.
"""
import os
import pickle
import time

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "scifit_chunks")
BM25_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "bm25_index.pkl"
)

RRF_K = 60  # constante standard pour Reciprocal Rank Fusion
RERANK_OVERFETCH = 3  # candidats bruts récupérés par chunk final gardé, avant reranking

# Singleton paresseux : le modèle de reranking (~80 Mo) n'est chargé qu'à la
# première utilisation réelle (rerank=True), pas à l'import du module ni
# pour les appelants qui n'activent jamais le reranking.
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> tuple[list[dict], int]:
    """Réordonne des chunks déjà récupérés avec un cross-encoder -- bien plus
    précis qu'un score de similarité bi-encoder ou lexical parce qu'il
    encode la question ET le chunk ensemble (attention croisée), mais trop
    lent pour scorer tout le corpus, d'où l'usage en second passage sur une
    petite short-list plutôt qu'en recherche primaire.

    Fonctionne sur n'importe quelle liste de dicts avec une clé "content" --
    réutilisable aussi bien pour les modes vector/bm25/hybrid que pour les
    relations du mode "graph" (voir streamlit_app/app.py).

    Retourne (chunks triés et tronqués à top_k, temps écoulé en ms).
    """
    if not chunks:
        return chunks, 0
    t0 = time.time()
    reranker = get_reranker()
    pairs = [[query, c["content"]] for c in chunks]
    scores = reranker.predict(pairs)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
    rerank_ms = int((time.time() - t0) * 1000)
    return ranked, rerank_ms


class HybridRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_chunks = bm25_data["chunks"]
        # Timing du dernier appel à search() : {"embedding_ms", "search_ms", "rerank_ms"}.
        # Alimente le panel de latence par étape dans Grafana (voir
        # streamlit_app/app.py et docs/monitoring.md) sans changer la
        # signature de search(), pour ne rien casser chez les appelants
        # existants (eval scripts).
        self.last_timing = {"embedding_ms": 0, "search_ms": 0, "rerank_ms": 0}

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        t0 = time.time()
        vec = self.model.encode([query])[0].tolist()
        self.last_timing["embedding_ms"] += int((time.time() - t0) * 1000)
        hits = self.qdrant.search(
            collection_name=QDRANT_COLLECTION, query_vector=vec, limit=top_k
        )
        return [hit.payload | {"score": hit.score} for hit in hits]

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.bm25_chunks[i] | {"score": float(scores[i])} for i in ranked_idx]

    def search(
        self, query: str, top_k: int = 8, mode: str = "hybrid", rerank: bool = False
    ) -> list[dict]:
        """
        mode: "vector" | "bm25" | "hybrid"
        rerank: si True, récupère top_k * RERANK_OVERFETCH candidats bruts
            puis les réordonne avec un cross-encoder avant de tronquer au
            top_k final (voir rerank_chunks()).
        Retourne une liste de chunks triés par pertinence, dédupliqués par chunk_id.
        """
        self.last_timing = {"embedding_ms": 0, "search_ms": 0, "rerank_ms": 0}
        t_start = time.time()
        fetch_k = top_k * RERANK_OVERFETCH if rerank else top_k

        if mode == "vector":
            candidates = self._vector_search(query, fetch_k)
        elif mode == "bm25":
            candidates = self._bm25_search(query, fetch_k)
        else:
            # hybride : RRF sur les deux classements
            vector_results = self._vector_search(query, fetch_k * 2)
            bm25_results = self._bm25_search(query, fetch_k * 2)

            rrf_scores: dict[str, float] = {}
            chunk_lookup: dict[str, dict] = {}

            for rank, chunk in enumerate(vector_results):
                cid = chunk["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
                chunk_lookup[cid] = chunk

            for rank, chunk in enumerate(bm25_results):
                cid = chunk["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
                chunk_lookup.setdefault(cid, chunk)

            ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:fetch_k]
            candidates = [chunk_lookup[cid] | {"score": rrf_scores[cid]} for cid in ranked_ids]

        elapsed_ms = int((time.time() - t_start) * 1000)
        self.last_timing["search_ms"] = max(0, elapsed_ms - self.last_timing["embedding_ms"])

        if rerank:
            candidates, rerank_ms = rerank_chunks(query, candidates, top_k)
            self.last_timing["rerank_ms"] = rerank_ms
            return candidates

        return candidates[:top_k]
