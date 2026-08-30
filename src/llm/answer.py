"""
Génère une réponse à partir des chunks récupérés (hybride et/ou graphe),
en forçant le LLM à citer ses sources et à indiquer un niveau de preuve
global, pour limiter l'hallucination et faciliter l'évaluation de
"faithfulness".
"""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

ANSWER_PROMPT = """Tu es un assistant scientifique spécialisé en nutrition et
performance sportive. Réponds à la question UNIQUEMENT à partir des extraits
fournis ci-dessous. Si les extraits ne permettent pas de répondre, dis-le
explicitement plutôt que d'inventer.

Pour chaque affirmation, indique la source entre crochets, ex: [source: 1].
Termine par une ligne "Niveau de preuve global: <faible|moyen|élevé>" basée
sur le nombre et la qualité (méta-analyse > essai randomisé > observationnelle)
des sources concordantes.

Question : {question}

Extraits disponibles :
{context}
"""


def format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        study_type = c.get("study_type", "unknown")
        year = c.get("year", "?")
        lines.append(
            f"[{i}] (type: {study_type}, année: {year}) {c['content']}"
        )
    return "\n\n".join(lines)


def generate_answer(question: str, chunks: list[dict]) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    context = format_context(chunks)
    prompt = ANSWER_PROMPT.format(question=question, context=context)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content
