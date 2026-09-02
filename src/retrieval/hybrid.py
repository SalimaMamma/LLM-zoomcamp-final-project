"""
Retrieval hybride : fusionne les résultats vecteur (Qdrant) et lexical
(BM25) avec Reciprocal Rank Fusion (RRF), une méthode simple et robuste
pour combiner deux classements sans avoir à calibrer des poids.
"""
import os
import pickle
import time

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "scifit_chunks")
BM25_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "bm25_index.pkl"
)

RRF_K = 60  # constante standard pour Reciprocal Rank Fusion


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
        # Timing du dernier appel à search() : {"embedding_ms": int, "search_ms": int}.
        # Alimente le panel de latence par étape dans Grafana (voir
        # streamlit_app/app.py et docs/monitoring.md) sans changer la
        # signature de search(), pour ne rien casser chez les appelants
        # existants (eval scripts).
        self.last_timing = {"embedding_ms": 0, "search_ms": 0}

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

    def search(self, query: str, top_k: int = 8, mode: str = "hybrid") -> list[dict]:
        """
        mode: "vector" | "bm25" | "hybrid"
        Retourne une liste de chunks triés par pertinence, dédupliqués par chunk_id.
        """
        self.last_timing = {"embedding_ms": 0, "search_ms": 0}
        t_start = time.time()

        if mode == "vector":
            results = self._vector_search(query, top_k)
            elapsed_ms = int((time.time() - t_start) * 1000)
            self.last_timing["search_ms"] = max(0, elapsed_ms - self.last_timing["embedding_ms"])
            return results

        if mode == "bm25":
            results = self._bm25_search(query, top_k)
            self.last_timing["search_ms"] = int((time.time() - t_start) * 1000)
            return results

        # hybride : RRF sur les deux classements
        vector_results = self._vector_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)

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

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        elapsed_ms = int((time.time() - t_start) * 1000)
        self.last_timing["search_ms"] = max(0, elapsed_ms - self.last_timing["embedding_ms"])
        return [chunk_lookup[cid] | {"score": rrf_scores[cid]} for cid in ranked_ids]
