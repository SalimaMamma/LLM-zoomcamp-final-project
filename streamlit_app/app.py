"""
Interface Streamlit — SciFit-Check

Pose une question sur nutrition/sport, choisis le mode de retrieval,
obtiens une réponse sourcée avec niveau de preuve, et donne un feedback.
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
from answer import generate_answer  # noqa: E402

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
        st.warning(f"Impossible d'enregistrer le feedback : {e}")


st.title("🏃 SciFit-Check")
st.caption(
    "Vérifie les idées reçues sur la nutrition et la performance sportive "
    "à partir de la littérature scientifique."
)

mode = st.radio(
    "Mode de retrieval",
    options=["hybrid", "vector", "bm25", "graph"],
    horizontal=True,
    help="hybrid = vecteur + BM25 (RRF) — graph = traversée du graphe de connaissances",
)

question = st.text_input(
    "Pose ta question",
    placeholder="Ex: Le jeûne intermittent améliore-t-il la performance en endurance ?",
)

if st.button("Vérifier", type="primary") and question:
    start = time.time()
    with st.spinner("Recherche dans la littérature scientifique..."):
        if mode == "graph":
            kg = load_graph()
            relations = kg.get_relations_for_entity(question)
            chunks = [
                {
                    "content": f"{r['subject']} {r['relation']} {r['object']}",
                    "study_type": "graph_relation",
                    "year": "-",
                }
                for r in relations
            ]
        else:
            retriever = load_retriever()
            chunks = retriever.search(question, top_k=6, mode=mode)

        if not chunks:
            st.warning("Aucune source pertinente trouvée. Essaie de reformuler ta question.")
        else:
            answer = generate_answer(question, chunks)
            latency_ms = int((time.time() - start) * 1000)

            st.markdown("### Réponse")
            st.markdown(answer)

            with st.expander(f"Sources utilisées ({len(chunks)})"):
                for i, c in enumerate(chunks, start=1):
                    st.markdown(
                        f"**[{i}]** *(type: {c.get('study_type', '?')}, "
                        f"année: {c.get('year', '?')})* — {c['content'][:300]}..."
                    )

            st.session_state["last_question"] = question
            st.session_state["last_answer"] = answer
            st.session_state["last_mode"] = mode
            st.session_state["last_latency"] = latency_ms

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Utile"):
                    log_feedback(question, answer, 1, mode, latency_ms)
                    st.success("Merci pour ton retour !")
            with col2:
                if st.button("👎 Pas utile"):
                    log_feedback(question, answer, -1, mode, latency_ms)
                    st.success("Merci pour ton retour !")

st.divider()
st.caption(
    "⚠️ Cet outil s'appuie sur des abstracts scientifiques et ne remplace pas "
    "un avis médical ou nutritionnel professionnel."
)
