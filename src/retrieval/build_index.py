"""
Construit les deux index de retrieval à partir des papiers stockés en
Postgres :
  1. Index vectoriel Qdrant (embeddings bge-small, local, gratuit)
  2. Index BM25 (rank_bm25), sérialisé sur disque pour le retrieval hybride

Écrit aussi les chunks dans la table `chunks` de Postgres pour traçabilité.

Usage:
    python src/retrieval/build_index.py
"""
import os
import pickle

import psycopg2
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from chunking import chunk_paper

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "scifit_chunks")
BM25_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "bm25_index.pkl"
)


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "scifit"),
        user=os.getenv("POSTGRES_USER", "scifit"),
        password=os.getenv("POSTGRES_PASSWORD", "scifit_local_pw"),
    )


def load_papers(cur) -> list[dict]:
    cur.execute("SELECT paper_id, abstract, study_type, year FROM papers WHERE abstract IS NOT NULL")
    rows = cur.fetchall()
    return [
        {"paper_id": r[0], "abstract": r[1], "study_type": r[2], "year": r[3]}
        for r in rows
    ]


def persist_chunks(cur, chunks: list[dict]):
    for c in chunks:
        cur.execute(
            """
            INSERT INTO chunks (chunk_id, paper_id, section, content)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            (c["chunk_id"], c["paper_id"], c["section"], c["content"]),
        )


def build_vector_index(all_chunks: list[dict]):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )

    sample_vec = model.encode(["test"])[0]
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=len(sample_vec), distance=Distance.COSINE),
    )

    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexation Qdrant"):
        batch = all_chunks[i : i + batch_size]
        vectors = model.encode([c["content"] for c in batch]).tolist()
        points = [
            PointStruct(
                id=idx + i,
                vector=vec,
                payload={
                    "chunk_id": c["chunk_id"],
                    "paper_id": c["paper_id"],
                    "section": c["section"],
                    "content": c["content"],
                    "study_type": c.get("study_type"),
                    "year": c.get("year"),
                },
            )
            for idx, (c, vec) in enumerate(zip(batch, vectors))
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def build_bm25_index(all_chunks: list[dict]):
    tokenized = [c["content"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)


def main():
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    papers = load_papers(cur)
    print(f"{len(papers)} papiers chargés depuis Postgres.")

    all_chunks = []
    for paper in tqdm(papers, desc="Chunking"):
        chunks = chunk_paper(paper["paper_id"], paper["abstract"])
        for c in chunks:
            c["study_type"] = paper["study_type"]
            c["year"] = paper["year"]
        persist_chunks(cur, chunks)
        all_chunks.extend(chunks)

    print(f"{len(all_chunks)} chunks générés.")

    build_vector_index(all_chunks)
    build_bm25_index(all_chunks)

    cur.close()
    conn.close()
    print("Indexation terminée (Qdrant + BM25).")


if __name__ == "__main__":
    main()
