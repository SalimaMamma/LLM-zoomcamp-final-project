"""
Génère un jeu de questions/réponses "vérité terrain" pour l'évaluation du
retrieval, sans annotation manuelle : un LLM (Groq) lit un chunk déjà
indexé et invente une question dont la réponse s'y trouve. Contrairement
au set annoté à la main (questions_annotees.json, mots-clés attendus
matchés par sous-chaîne), chaque question ici est directement rattachée
au chunk_id/paper_id exact dont elle a été générée -- un vrai gold label,
pas un proxy approximatif. Voir docs/evaluation.md.

Usage:
    python src/eval/generate_synthetic_questions.py
"""
import json
import os
import random

import psycopg2
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()

OUT_PATH = os.path.join(os.path.dirname(__file__), "synthetic_questions.json")
N_CHUNKS = 18  # ~2 questions/chunk -> ~30-36 questions, dans la fourchette visée
QUESTIONS_PER_CHUNK = 2
MIN_CONTENT_LEN = 150  # évite les chunks trop courts pour donner une vraie question

GENERATION_PROMPT = """You are building a test set to evaluate a retrieval
system. Given the following excerpt from a scientific abstract about
sports nutrition or athletic performance, write exactly {n} distinct,
specific questions whose answer is directly and ONLY stated in this
excerpt (not answerable from general knowledge). Phrase each question the
way a curious person would naturally ask it -- not "what does this text
say about...". For each question, also give a short one-sentence answer
using only information from the excerpt.

Reply ONLY with JSON of the form:
{{"pairs": [{{"question": "...", "answer": "..."}}, ...]}}

Excerpt:
\"\"\"{content}\"\"\"
"""


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "scifit"),
        user=os.getenv("POSTGRES_USER", "scifit"),
        password=os.getenv("POSTGRES_PASSWORD", "scifit_local_pw"),
    )


def sample_chunks(cur, n: int) -> list[dict]:
    """Échantillonne des chunks result/conclusion (contiennent des
    affirmations factuelles concrètes, contrairement à background/method),
    un seul par paper_id pour ne pas sur-représenter un article."""
    cur.execute(
        """
        SELECT DISTINCT ON (paper_id) chunk_id, paper_id, content, section
        FROM chunks
        WHERE section IN ('result', 'conclusion') AND length(content) > %s
        ORDER BY paper_id, random()
        """,
        (MIN_CONTENT_LEN,),
    )
    rows = cur.fetchall()
    random.shuffle(rows)
    return [
        {"chunk_id": r[0], "paper_id": r[1], "content": r[2], "section": r[3]}
        for r in rows[:n]
    ]


def generate_pairs(client: Groq, model: str, content: str) -> list[dict]:
    prompt = GENERATION_PROMPT.format(n=QUESTIONS_PER_CHUNK, content=content)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        return parsed.get("pairs", [])
    except Exception as e:
        print(f"[WARN] génération échouée: {e}")
        return []


def main():
    conn = get_connection()
    cur = conn.cursor()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    chunks = sample_chunks(cur, N_CHUNKS)
    print(f"{len(chunks)} chunks échantillonnés (1 par papier, sections result/conclusion).")

    dataset = []
    for c in tqdm(chunks, desc="Génération Q/A"):
        pairs = generate_pairs(client, model, c["content"])
        for p in pairs:
            if not p.get("question") or not p.get("answer"):
                continue
            dataset.append(
                {
                    "question": p["question"],
                    "answer": p["answer"],
                    "gold_chunk_id": c["chunk_id"],
                    "gold_paper_id": c["paper_id"],
                    "section": c["section"],
                }
            )

    cur.close()
    conn.close()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n{len(dataset)} paires question/réponse générées -> {OUT_PATH}")
    print("Pas de filtrage manuel dans ce script (voir docs/evaluation.md) --")
    print("relire le fichier avant de s'y fier pour un jugement définitif.")


if __name__ == "__main__":
    main()
