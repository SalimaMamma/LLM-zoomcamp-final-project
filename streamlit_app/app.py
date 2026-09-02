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

from hybrid import HybridRetriever  # noqa: E402
from graph_store import KnowledgeGraph  # noqa: E402
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
                question, retrieval_mode, embedding_ms, search_ms, generation_ms,
                prompt_tokens, completion_tokens, avg_chunk_score, min_chunk_score,
                chunk_count, paper_ids, answer_length, is_declined, is_well_formatted,
                is_error, error_message
            ) VALUES (
                %(question)s, %(retrieval_mode)s, %(embedding_ms)s, %(search_ms)s, %(generation_ms)s,
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

question = st.text_input(
    "Ask your question",
    placeholder="E.g.: Does intermittent fasting improve endurance performance?",
)

if st.button("Check", type="primary") and question:
    start = time.time()
    with st.spinner("Searching the scientific literature..."):
        embedding_ms, search_ms = 0, 0
        if mode == "graph":
            kg = load_graph()
            t0 = time.time()
            relations = kg.get_relations_for_entity(question)
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
        else:
            retriever = load_retriever()
            chunks = retriever.search(question, top_k=6, mode=mode)
            embedding_ms = retriever.last_timing["embedding_ms"]
            search_ms = retriever.last_timing["search_ms"]

        scores = [c["score"] for c in chunks if "score" in c]
        paper_ids = list({c["paper_id"] for c in chunks if c.get("paper_id")})
        common = dict(
            question=question,
            retrieval_mode=mode,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
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
