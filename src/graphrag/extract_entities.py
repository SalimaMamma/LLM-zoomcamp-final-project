"""
Extraction d'entités et de relations depuis les abstracts pour construire
un graphe de connaissances léger (NetworkX + persistance Postgres).

Chaque abstract est passé au LLM (Groq) avec une consigne stricte de
sortie JSON, pour extraire des triplets (sujet, relation, objet), par
exemple :
    ("intermittent fasting", "no_effect_on", "endurance performance")
    ("meta-analysis X", "contradicts", "study Y")

Usage:
    python src/graphrag/extract_entities.py
"""
import json
import os

import psycopg2
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()

EXTRACTION_PROMPT = """Tu es un assistant d'extraction d'information scientifique.
À partir de l'abstract suivant, extrais entre 1 et 5 triplets
(sujet, relation, objet) qui capturent les affirmations factuelles
principales (interventions, effets, contradictions avec d'autres travaux).

Relations autorisées : improves, reduces, no_effect_on, increases,
decreases, contradicts, confirms, associated_with.

Réponds UNIQUEMENT avec un JSON de la forme :
{{"triplets": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Abstract :
\"\"\"{abstract}\"\"\"
"""


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "scifit"),
        user=os.getenv("POSTGRES_USER", "scifit"),
        password=os.getenv("POSTGRES_PASSWORD", "scifit_local_pw"),
    )


def extract_triplets(client: Groq, abstract: str) -> list[dict]:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(abstract=abstract)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        return parsed.get("triplets", [])
    except Exception as e:
        print(f"[WARN] extraction échouée: {e}")
        return []


def main():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SELECT paper_id, abstract FROM papers WHERE abstract IS NOT NULL")
    papers = cur.fetchall()

    total_edges = 0
    for paper_id, abstract in tqdm(papers, desc="Extraction graphe"):
        # évite de ré-extraire si déjà fait
        cur.execute("SELECT 1 FROM graph_edges WHERE source_paper_id = %s LIMIT 1", (paper_id,))
        if cur.fetchone():
            continue

        triplets = extract_triplets(client, abstract)
        for t in triplets:
            if not all(k in t for k in ("subject", "relation", "object")):
                continue
            cur.execute(
                """
                INSERT INTO graph_edges (source_paper_id, subject, relation, object)
                VALUES (%s, %s, %s, %s)
                """,
                (paper_id, t["subject"].lower(), t["relation"].lower(), t["object"].lower()),
            )
            total_edges += 1

    cur.close()
    conn.close()
    print(f"Extraction terminée : {total_edges} relations ajoutées au graphe.")


if __name__ == "__main__":
    main()
