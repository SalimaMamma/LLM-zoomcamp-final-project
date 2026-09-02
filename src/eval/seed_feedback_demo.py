"""
Script de démo/seed : fait tourner le pipeline réel (retrieval + génération)
sur les questions annotées, à travers les différents modes, et enregistre le
résultat dans la table `feedback` exactement comme le fait l'app Streamlit
(streamlit_app/app.py::log_feedback), pour avoir des données réalistes à
montrer sur le dashboard Grafana.

Ce n'est PAS un générateur de fausses données : chaque ligne vient d'un vrai
appel au retriever hybride + au LLM Groq. Le rating (👍/👎) n'est pas un
avis humain réel (aucun humain n'a jugé ces réponses) — il est dérivé d'une
vérification automatique et déterministe du format attendu (citation d'au
moins une source + ligne "Niveau de preuve global" présente, cf. prompt
dans src/llm/answer.py). Ce script sert à peupler le dashboard Grafana pour
la démo ; en usage réel, le rating vient des clics 👍/👎 des utilisateurs
dans l'app Streamlit.

Usage:
    python src/eval/seed_feedback_demo.py
"""
import json
import os
import re
import sys
import time

import psycopg2
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "graphrag"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "llm"))

from hybrid import HybridRetriever  # noqa: E402
from graph_store import KnowledgeGraph  # noqa: E402
from answer import generate_answer  # noqa: E402

load_dotenv()

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions_annotees.json")
MODES = ["hybrid", "vector", "bm25", "graph"]


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "scifit"),
        user=os.getenv("POSTGRES_USER", "scifit"),
        password=os.getenv("POSTGRES_PASSWORD", "scifit_local_pw"),
    )


NO_EVIDENCE_MARKERS = (
    "ne contiennent aucune information",
    "ne permettent pas de répondre",
    "ne traite du",
    "ne traitent du",
    "aucun des extraits",
    "aucune donnée",
)


def format_compliant_rating(answer: str) -> int:
    """Vérification automatique et déterministe (pas un jugement humain) :

    - la réponse cite au moins une source (le prompt demande "[source: n]",
      mais le modèle utilise parfois juste "[n]" -> les deux comptent) ET
      affiche la ligne de niveau de preuve -> positif ;
    - la réponse décline explicitement faute de source pertinente (comportement
      voulu par le prompt : "dis-le explicitement plutôt que d'inventer") ->
      positif aussi, ce n'est pas un échec de format, c'est le comportement
      demandé ;
    - sinon -> négatif (réponse qui n'affiche ni citation, ni aveu d'absence
      de source, ni niveau de preuve -> signal de mauvais format).
    """
    lowered = answer.lower()
    if any(marker in lowered for marker in NO_EVIDENCE_MARKERS):
        return 1
    has_citation = bool(re.search(r"\[(?:source\s*:\s*)?\d+\]", answer, re.IGNORECASE))
    has_evidence_line = "niveau de preuve" in lowered
    return 1 if (has_citation and has_evidence_line) else -1


def log_feedback(cur, question, answer, rating, mode, latency_ms):
    cur.execute(
        """
        INSERT INTO feedback (question, answer, rating, retrieval_mode, latency_ms)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (question, answer, rating, mode, latency_ms),
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
            start = time.time()
            if mode == "graph":
                relations = kg.get_relations_for_entity(q["expected_keywords"][0])
                chunks = [
                    {"content": f"{r['subject']} {r['relation']} {r['object']}",
                     "study_type": "graph_relation", "year": "-"}
                    for r in relations
                ]
            else:
                chunks = retriever.search(q["question"], top_k=6, mode=mode)

            if not chunks:
                continue

            try:
                answer = generate_answer(q["question"], chunks)
            except Exception as e:
                # rate limit (TPM/TPD) ou autre erreur Groq -> on saute cette
                # interaction plutôt que de perdre tout le seeding déjà fait
                print(f"[{mode:6s}] {q['question'][:50]:50s} -> ÉCHOUÉ ({e})")
                continue
            latency_ms = int((time.time() - start) * 1000)
            rating = format_compliant_rating(answer)

            log_feedback(cur, q["question"], answer, rating, mode, latency_ms)
            total += 1
            print(f"[{mode:6s}] {q['question'][:50]:50s} -> {rating:+d} ({latency_ms}ms)")

            # Le tier gratuit Groq limite aussi les tokens/minute (pas
            # seulement /jour) -> on laisse respirer entre deux appels pour
            # ne pas se faire jeter en plein milieu du seeding.
            time.sleep(8)

    cur.close()
    conn.close()
    print(f"\n{total} interactions enregistrées dans la table feedback.")


if __name__ == "__main__":
    main()
