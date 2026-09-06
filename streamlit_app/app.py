"""
SciFit-Check — Streamlit interface

Ask a sports nutrition / performance question, pick a retrieval mode, get a
sourced answer with an evidence level, and leave feedback.

Every request (not just ones with explicit 👍/👎 feedback) is logged to
`retrieval_log` for the Grafana monitoring dashboard — latency per stage,
token cost, retrieval score quality, source diversity, error rate. See
docs/monitoring.md.
"""
import os
import sys
import time

import psycopg2
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "graphrag"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "llm"))

from hybrid import HybridRetriever, rerank_chunks  # noqa: E402
from graph_store import KnowledgeGraph, get_papers_metadata  # noqa: E402
from answer import (  # noqa: E402
    generate_answer_with_usage,
    is_declined_answer,
    is_well_formatted_answer,
)

load_dotenv()

st.set_page_config(page_title="SciFit-Check", page_icon="🏃", layout="centered")


@st.cache_resource
def load_retriever():
    return HybridRetriever()


@st.cache_resource
def load_graph():
    return KnowledgeGraph()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "scifit"),
        user=os.getenv("POSTGRES_USER", "scifit"),
        password=os.getenv("POSTGRES_PASSWORD", "scifit_local_pw"),
    )


def log_feedback(question: str, answer: str, rating: int, mode: str, latency_ms: int):
    try:
        conn = get_connection()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback (question, answer, rating, retrieval_mode, latency_ms)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (question, answer, rating, mode, latency_ms),
        )
        cur.close()
        conn.close()
    except Exception as e:
        st.warning(f"Couldn't save feedback: {e}")


def log_retrieval(**kw):
    """Log every request (success, decline, or error) for the monitoring
    dashboard — not just the ones a user happens to rate."""
    try:
        conn = get_connection()
        conn.autocommit = True
        cur = conn.cursor()
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
        cur.close()
        conn.close()
    except Exception:
        pass  # le monitoring ne doit jamais faire planter une vraie requête utilisateur


st.title("🏃 SciFit-Check")
st.caption(
    "Fact-check sports nutrition and performance claims against the "
    "scientific literature."
)

mode = st.radio(
    "Retrieval mode",
    options=["hybrid", "vector", "bm25", "graph"],
    horizontal=True,
    help="hybrid = vector + BM25 (RRF) — graph = knowledge graph traversal",
)

rerank = st.checkbox(
    "🎯 Re-rank with cross-encoder",
    value=True,
    help=(
        "Second pass with a cross-encoder (ms-marco-MiniLM) over a larger "
        "candidate pool before picking the final sources. On by default: "
        "on a 24-question gold-labeled evaluation, it improves every mode, "
        "including hybrid (chunk recall@8 0.83→0.92, MRR 0.74→0.90) — see "
        "docs/evaluation.md for the full comparison, including an earlier, "
        "smaller evaluation that pointed the other way."
    ),
)

question = st.text_input(
    "Ask your question",
    placeholder="E.g.: Does intermittent fasting improve endurance performance?",
)

if st.button("Check", type="primary") and question:
    start = time.time()
    with st.spinner("Searching the scientific literature..."):
        embedding_ms, search_ms, rerank_ms = 0, 0, 0
        try:
            if mode == "graph":
                kg = load_graph()
                t0 = time.time()
                relations = kg.get_relations_for_entity(question)
                search_ms = int((time.time() - t0) * 1000)
                # Un terme courant ("performance") peut matcher des dizaines de
                # nœuds et retourner des centaines de relations -- sans borne,
                # ça part tout dans le prompt du LLM (vu en pratique : 107
                # chunks, 2500+ tokens, ~2x un appel normal). On plafonne comme
                # les autres modes, avec plus de marge si le reranking va
                # ensuite trier ce pool pour ne garder que le meilleur top_k.
                pool_size = 18 if rerank else 6
                relations = relations[:pool_size]
                # Une relation seule ("caffeine improves performance") n'est pas
                # traçable : on va rechercher le papier source (titre + URL +
                # vrai type d'étude/année) pour que la citation pointe vers un
                # article réel, pas juste un triplet flottant.
                paper_meta = get_papers_metadata([r.get("paper_id") for r in relations])
                chunks = [
                    {
                        "content": f"{r['subject']} {r['relation']} {r['object']}",
                        "study_type": paper_meta.get(r.get("paper_id"), {}).get("study_type") or "unknown",
                        "year": paper_meta.get(r.get("paper_id"), {}).get("year") or "?",
                        "paper_id": r.get("paper_id"),
                        "paper_title": paper_meta.get(r.get("paper_id"), {}).get("title"),
                        "paper_url": paper_meta.get(r.get("paper_id"), {}).get("url"),
                    }
                    for r in relations
                ]
                if rerank:
                    chunks, rerank_ms = rerank_chunks(question, chunks, top_k=6)
            else:
                retriever = load_retriever()
                chunks = retriever.search(question, top_k=6, mode=mode, rerank=rerank)
                embedding_ms = retriever.last_timing["embedding_ms"]
                search_ms = retriever.last_timing["search_ms"]
                rerank_ms = retriever.last_timing["rerank_ms"]
        except Exception as e:
            # Qdrant/BM25/reranker peuvent aussi échouer (timeout réseau
            # transitoire vu en pratique) -- pas seulement Groq côté génération.
            st.error(f"Search failed: {e}")
            log_retrieval(
                question=question, retrieval_mode=mode, embedding_ms=None, search_ms=None,
                rerank_ms=None, reranked=rerank, generation_ms=None, prompt_tokens=None,
                completion_tokens=None, avg_chunk_score=None, min_chunk_score=None,
                chunk_count=None, paper_ids=None, answer_length=None, is_declined=None,
                is_well_formatted=None, is_error=True, error_message=str(e)[:500],
            )
            st.stop()

        scores = [c["score"] for c in chunks if "score" in c]
        paper_ids = list({c["paper_id"] for c in chunks if c.get("paper_id")})
        common = dict(
            question=question,
            retrieval_mode=mode,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
            reranked=rerank,
            avg_chunk_score=sum(scores) / len(scores) if scores else None,
            min_chunk_score=min(scores) if scores else None,
            chunk_count=len(chunks),
            paper_ids=paper_ids or None,
        )

        if not chunks:
            st.warning("No relevant source found. Try rephrasing your question.")
            log_retrieval(
                **common, generation_ms=None, prompt_tokens=None, completion_tokens=None,
                answer_length=None, is_declined=None, is_well_formatted=None,
                is_error=False, error_message=None,
            )
        else:
            try:
                answer, usage, generation_ms = generate_answer_with_usage(question, chunks)
            except Exception as e:
                st.error(f"Answer generation failed: {e}")
                log_retrieval(
                    **common, generation_ms=None, prompt_tokens=None, completion_tokens=None,
                    answer_length=None, is_declined=None, is_well_formatted=None,
                    is_error=True, error_message=str(e)[:500],
                )
                st.stop()

            latency_ms = int((time.time() - start) * 1000)
            log_retrieval(
                **common,
                generation_ms=generation_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                answer_length=len(answer),
                is_declined=is_declined_answer(answer),
                is_well_formatted=is_well_formatted_answer(answer),
                is_error=False,
                error_message=None,
            )

            st.markdown("### Answer")
            st.markdown(answer)

            with st.expander(f"Sources used ({len(chunks)})"):
                for i, c in enumerate(chunks, start=1):
                    st.markdown(
                        f"**[{i}]** *(type: {c.get('study_type', '?')}, "
                        f"year: {c.get('year', '?')})* — {c['content'][:300]}..."
                    )
                    if c.get("paper_title"):
                        # Mode graph : la relation seule (sujet/relation/objet)
                        # ne dit pas d'où elle vient -- on relie explicitement
                        # au papier source dont elle a été extraite.
                        label = c["paper_title"]
                        ref = f"[{label}]({c['paper_url']})" if c.get("paper_url") else label
                        st.caption(f"↳ extracted from: {ref}")

            st.session_state["last_question"] = question
            st.session_state["last_answer"] = answer
            st.session_state["last_mode"] = mode
            st.session_state["last_latency"] = latency_ms

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Useful"):
                    log_feedback(question, answer, 1, mode, latency_ms)
                    st.success("Thanks for the feedback!")
            with col2:
                if st.button("👎 Not useful"):
                    log_feedback(question, answer, -1, mode, latency_ms)
                    st.success("Thanks for the feedback!")

st.divider()
st.caption(
    "⚠️ This tool relies on scientific abstracts and does not replace "
    "professional medical or nutritional advice."
)
