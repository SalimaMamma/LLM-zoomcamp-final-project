"""
Script de démo/seed : fait tourner le pipeline réel (retrieval + génération)
sur les questions annotées, à travers les différents modes, et enregistre le
résultat dans `feedback` ET `retrieval_log` exactement comme le fait l'app
Streamlit (streamlit_app/app.py), pour avoir des données réalistes à montrer
sur le dashboard Grafana (feedback simulé + observabilité : latence par
étape, tokens, scores de retrieval, diversité des sources...).

Ce n'est PAS un générateur de fausses données : chaque ligne vient d'un vrai
appel au retriever hybride + au LLM Groq. Le rating (👍/👎) n'est pas un
avis humain réel (aucun humain n'a jugé ces réponses) — il est dérivé des
mêmes vérifications automatiques et déterministes que l'app utilise pour son
propre monitoring (src/llm/answer.py::is_declined_answer /
is_well_formatted_answer). En usage réel, le rating vient des clics 👍/👎
des utilisateurs dans l'app Streamlit.

Usage:
    python src/eval/seed_feedback_demo.py
"""
import json
import os
import sys
import time

import psycopg2
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "graphrag"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "llm"))

from hybrid import HybridRetriever, rerank_chunks  # noqa: E402
from graph_store import KnowledgeGraph  # noqa: E402
from answer import (  # noqa: E402
    generate_answer_with_usage,
    is_declined_answer,
    is_well_formatted_answer,
)

load_dotenv()

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions_annotees.json")
MODES = ["hybrid", "vector", "bm25", "graph"]
# L'app désactive le reranking par défaut (voir docs/evaluation.md -- ça
# aide bm25 mais dégrade hybrid sur l'éval mesurée). Ici on le force à True
# quand même, uniquement pour peupler les colonnes rerank_ms/reranked du
# dashboard de démo avec de vraies données plutôt que des NULL partout.
RERANK = True


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "scifit"),
        user=os.getenv("POSTGRES_USER", "scifit"),
        password=os.getenv("POSTGRES_PASSWORD", "scifit_local_pw"),
    )


def rating_from_answer(answer: str) -> int:
    """Positif si bien formatée OU si elle décline explicitement faute de
    source (comportement voulu, pas un échec) ; négatif sinon."""
    if is_declined_answer(answer) or is_well_formatted_answer(answer):
        return 1
    return -1


def log_feedback(cur, question, answer, rating, mode, latency_ms):
    cur.execute(
        """
        INSERT INTO feedback (question, answer, rating, retrieval_mode, latency_ms)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (question, answer, rating, mode, latency_ms),
    )


def log_retrieval(cur, **kw):
    cur.execute(
        """
        INSERT INTO retrieval_log (
            question, retrieval_mode, embedding_ms, search_ms, rerank_ms, reranked, generation_ms,
            prompt_tokens, completion_tokens, avg_chunk_score, min_chunk_score,
            chunk_count, paper_ids, answer_length, is_declined, is_well_formatted,
            is_error, error_message
        ) VALUES (
            %(question)s, %(retrieval_mode)s, %(embedding_ms)s, %(search_ms)s, %(rerank_ms)s, %(reranked)s, %(generation_ms)s,
            %(prompt_tokens)s, %(completion_tokens)s, %(avg_chunk_score)s, %(min_chunk_score)s,
            %(chunk_count)s, %(paper_ids)s, %(answer_length)s, %(is_declined)s, %(is_well_formatted)s,
            %(is_error)s, %(error_message)s
        )
        """,
        kw,
    )


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    retriever = HybridRetriever()
    kg = KnowledgeGraph()
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    total = 0
    for q in questions:
        for mode in MODES:
            embedding_ms, search_ms, rerank_ms = 0, 0, 0
            try:
                if mode == "graph":
                    t0 = time.time()
                    relations = kg.get_relations_for_entity(q["expected_keywords"][0])
                    search_ms = int((time.time() - t0) * 1000)
                    chunks = [
                        {
                            "content": f"{r['subject']} {r['relation']} {r['object']}",
                            "study_type": "graph_relation",
                            "year": "-",
                            "paper_id": r.get("paper_id"),
                        }
                        for r in relations
                    ]
                    if RERANK:
                        chunks, rerank_ms = rerank_chunks(q["question"], chunks, top_k=6)
                else:
                    chunks = retriever.search(q["question"], top_k=6, mode=mode, rerank=RERANK)
                    embedding_ms = retriever.last_timing["embedding_ms"]
                    search_ms = retriever.last_timing["search_ms"]
                    rerank_ms = retriever.last_timing["rerank_ms"]
            except Exception as e:
                # Qdrant/BM25/reranker peuvent aussi échouer (timeout réseau
                # transitoire vu en pratique) -- pas seulement Groq. On logue
                # et on continue plutôt que de perdre tout le seeding restant.
                cur.execute(
                    """
                    INSERT INTO retrieval_log (question, retrieval_mode, is_error, error_message)
                    VALUES (%s, %s, true, %s)
                    """,
                    (q["question"], mode, str(e)[:500]),
                )
                print(f"[{mode:6s}] {q['question'][:50]:50s} -> ÉCHOUÉ retrieval ({e})")
                continue

            scores = [c["score"] for c in chunks if "score" in c]
            paper_ids = list({c["paper_id"] for c in chunks if c.get("paper_id")})
            common = dict(
                question=q["question"],
                retrieval_mode=mode,
                embedding_ms=embedding_ms,
                search_ms=search_ms,
                rerank_ms=rerank_ms,
                reranked=RERANK,
                avg_chunk_score=sum(scores) / len(scores) if scores else None,
                min_chunk_score=min(scores) if scores else None,
                chunk_count=len(chunks),
                paper_ids=paper_ids or None,
            )

            if not chunks:
                log_retrieval(
                    cur, **common, generation_ms=None, prompt_tokens=None,
                    completion_tokens=None, answer_length=None, is_declined=None,
                    is_well_formatted=None, is_error=False, error_message=None,
                )
                print(f"[{mode:6s}] {q['question'][:50]:50s} -> pas de résultat (0 chunk)")
                continue

            try:
                answer, usage, generation_ms = generate_answer_with_usage(q["question"], chunks)
            except Exception as e:
                # rate limit (TPM/TPD) ou autre erreur Groq -> on logue l'échec
                # et on continue plutôt que de perdre tout le seeding déjà fait
                log_retrieval(
                    cur, **common, generation_ms=None, prompt_tokens=None,
                    completion_tokens=None, answer_length=None, is_declined=None,
                    is_well_formatted=None, is_error=True, error_message=str(e)[:500],
                )
                print(f"[{mode:6s}] {q['question'][:50]:50s} -> ÉCHOUÉ ({e})")
                continue

            rating = rating_from_answer(answer)
            latency_ms = embedding_ms + search_ms + rerank_ms + generation_ms

            log_feedback(cur, q["question"], answer, rating, mode, latency_ms)
            log_retrieval(
                cur, **common,
                generation_ms=generation_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                answer_length=len(answer),
                is_declined=is_declined_answer(answer),
                is_well_formatted=is_well_formatted_answer(answer),
                is_error=False,
                error_message=None,
            )
            total += 1
            print(f"[{mode:6s}] {q['question'][:50]:50s} -> {rating:+d} ({latency_ms}ms)")

            # Le tier gratuit Groq limite aussi les tokens/minute (pas
            # seulement /jour) -> on laisse respirer entre deux appels pour
            # ne pas se faire jeter en plein milieu du seeding.
            time.sleep(8)

    cur.close()
    conn.close()
    print(f"\n{total} interactions enregistrées dans feedback + retrieval_log.")


if __name__ == "__main__":
    main()
