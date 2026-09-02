"""
Génère une réponse à partir des chunks récupérés (hybride et/ou graphe),
en forçant le LLM à citer ses sources et à indiquer un niveau de preuve
global, pour limiter l'hallucination et faciliter l'évaluation de
"faithfulness".
"""
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

ANSWER_PROMPT = """You are a scientific assistant specialized in sports nutrition
and athletic performance. Answer the question ONLY using the excerpts provided
below. If the excerpts don't allow you to answer, say so explicitly rather
than making something up.

For every claim, cite the source in brackets, e.g. [source: 1].
End with a line "Overall evidence level: <low|medium|high>" based on the
number and quality (meta-analysis > randomized trial > observational) of
the concordant sources.

Question: {question}

Available excerpts:
{context}
"""

# Marqueurs utilisés pour détecter une réponse qui décline explicitement
# faute de source pertinente -- c'est le comportement VOULU par le prompt
# ci-dessus, pas un échec de format. Utilisé à la fois pour le monitoring
# live (streamlit_app/app.py) et le seed de démo (src/eval/seed_feedback_demo.py).
NO_EVIDENCE_MARKERS = (
    "don't contain any information",
    "do not contain any information",
    "doesn't allow",
    "does not allow",
    "none of the excerpts",
    "none of the provided excerpts",
    "no excerpt",
    "no data",
    "cannot be answered",
    "can't be answered",
    "not possible to answer",
)


def format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        study_type = c.get("study_type", "unknown")
        year = c.get("year", "?")
        lines.append(
            f"[{i}] (type: {study_type}, year: {year}) {c['content']}"
        )
    return "\n\n".join(lines)


def is_declined_answer(answer: str) -> bool:
    """La réponse décline explicitement faute de source pertinente (comportement
    demandé par le prompt : "say so explicitly rather than making something up")."""
    lowered = answer.lower()
    return any(marker in lowered for marker in NO_EVIDENCE_MARKERS)


def is_well_formatted_answer(answer: str) -> bool:
    """Citation d'au moins une source (le prompt demande "[source: n]", mais le
    modèle utilise parfois juste "[n]" -> les deux comptent) ET ligne de niveau
    de preuve présentes."""
    has_citation = bool(re.search(r"\[(?:source\s*:\s*)?\d+\]", answer, re.IGNORECASE))
    has_evidence_line = "evidence level" in answer.lower()
    return has_citation and has_evidence_line


def generate_answer_with_usage(question: str, chunks: list[dict]) -> tuple[str, dict, int]:
    """Comme generate_answer(), mais expose aussi les tokens consommés et le
    temps de génération -- utilisé par l'app pour le monitoring (coût par
    requête, latence par étape). Voir docs/monitoring.md."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    context = format_context(chunks)
    prompt = ANSWER_PROMPT.format(question=question, context=context)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    generation_ms = int((time.time() - t0) * 1000)

    answer = resp.choices[0].message.content
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
    }
    return answer, usage, generation_ms


def generate_answer(question: str, chunks: list[dict]) -> str:
    answer, _usage, _generation_ms = generate_answer_with_usage(question, chunks)
    return answer
