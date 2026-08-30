"""
Compare plusieurs stratégies de prompt pour la génération de réponse, en
utilisant un second appel LLM comme juge ("LLM-as-judge") pour scorer la
fidélité de la réponse aux sources récupérées (faithfulness) — c'est-à-dire
si chaque affirmation est bien ancrée dans un chunk fourni, et non
halluciné.

Usage:
    python src/eval/llm_eval.py
"""
import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from hybrid import HybridRetriever  # noqa: E402

load_dotenv()

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions_annotees.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval_results")

# Deux stratégies de prompt à comparer
PROMPT_STRATEGIES = {
    "zero_shot": (
        "Réponds à la question suivante uniquement à partir des extraits fournis. "
        "Cite tes sources entre crochets.\n\nQuestion: {question}\n\nExtraits:\n{context}"
    ),
    "structured_evidence": (
        "Tu es un assistant scientifique rigoureux. Réponds à la question en structurant "
        "ta réponse ainsi : 1) Résumé en une phrase 2) Preuves à l'appui (avec citations "
        "[source: n]) 3) Limites ou contradictions éventuelles. N'invente rien qui ne soit "
        "pas dans les extraits.\n\nQuestion: {question}\n\nExtraits:\n{context}"
    ),
}

JUDGE_PROMPT = """Tu es un évaluateur strict. Voici des extraits sources et une réponse
générée par un autre modèle. Note de 0 à 10 la fidélité de la réponse aux
sources (0 = contient des affirmations non supportées par les sources,
10 = chaque affirmation est directement traçable à une source).

Réponds UNIQUEMENT avec un JSON: {{"faithfulness_score": <0-10>, "justification": "..."}}

Extraits:
{context}

Réponse à évaluer:
{answer}
"""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {c['content']}" for i, c in enumerate(chunks, start=1))


def generate_with_strategy(client: Groq, model: str, strategy_key: str, question: str, context: str) -> str | None:
    prompt = PROMPT_STRATEGIES[strategy_key].format(question=question, context=context)
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2
        )
    except Exception as e:
        # Rate limit / erreur API -> on saute cette question plutôt que de
        # perdre tous les résultats déjà accumulés.
        print(f"[WARN] génération échouée ({strategy_key}): {e}")
        return None
    return resp.choices[0].message.content


def judge_faithfulness(client: Groq, model: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(context=context, answer=answer)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        # Le mode JSON du modèle échoue parfois à produire un JSON valide
        # (ex: guillemets typographiques non échappés) -> on ne bloque pas
        # toute l'évaluation pour une question.
        print(f"[WARN] juge échoué: {e}")
        return {"faithfulness_score": None, "justification": "judge_error"}
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"faithfulness_score": None, "justification": "parse_error"}


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    retriever = HybridRetriever()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    results = {key: [] for key in PROMPT_STRATEGIES}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "llm_eval.json")

    for q in questions:
        chunks = retriever.search(q["question"], top_k=6, mode="hybrid")
        context = format_context(chunks)

        for strategy_key in PROMPT_STRATEGIES:
            answer = generate_with_strategy(client, model, strategy_key, q["question"], context)
            if answer is None:
                continue  # appel échoué (ex: rate limit) -> on ne compte pas cette question
            judgment = judge_faithfulness(client, model, context, answer)
            results[strategy_key].append(
                {
                    "question": q["question"],
                    "answer": answer,
                    "faithfulness_score": judgment.get("faithfulness_score"),
                    "justification": judgment.get("justification"),
                }
            )
            # sauvegarde incrémentale : un rate limit en cours de route ne
            # doit pas faire perdre les résultats déjà obtenus
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("Score moyen de fidélité par stratégie :")
    for strategy_key, entries in results.items():
        scores = [e["faithfulness_score"] for e in entries if e["faithfulness_score"] is not None]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {strategy_key:22s} — {avg:.1f}/10")
    print(f"\nDétails sauvegardés dans {out_path}")


if __name__ == "__main__":
    main()
