"""
Ingestion incrémentale de papiers scientifiques via l'API OpenAlex.

OpenAlex est entièrement gratuit, ne nécessite AUCUNE clé API, et a un
rate limit généreux (jusqu'à 100 000 requêtes/jour dans le "polite pool"
si on renseigne un email de contact). C'est la source la plus fiable des
trois pour un projet type bootcamp — pas de 429 intempestifs comme sur
Semantic Scholar.

Doc: https://docs.openalex.org/api-entities/works

Usage:
    python src/ingestion/openalex.py
"""
import os
import re
import time

import psycopg2
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_URL = "https://api.openalex.org/works"

# Email de contact pour rejoindre le "polite pool" (rate limit bien plus élevé).
# Mets ton propre email dans .env (OPENALEX_MAILTO) ou laisse vide.
MAILTO = os.getenv("OPENALEX_MAILTO", "")

QUERIES = [
    "intermittent fasting endurance performance",
    "protein timing muscle protein synthesis",
    "carbohydrate loading marathon performance",
    "caffeine athletic performance",
    "creatine supplementation strength training",
    "post exercise nutrition recovery",
    "fasted cardio fat oxidation",
    "hydration strategy endurance exercise",
    "sports nutrition myths",
    "branched chain amino acids exercise",
]

RESULTS_PER_QUERY = 30

# OpenAlex classe les types de publication différemment de Semantic Scholar.
TYPE_TO_STUDY_TYPE = {
    "review": "meta-analysis",
    "article": "unknown",  # affiné plus bas via le titre/les concepts si besoin
}


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "scifit"),
        user=os.getenv("POSTGRES_USER", "scifit"),
        password=os.getenv("POSTGRES_PASSWORD", "scifit_local_pw"),
    )


def clean_text(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"\s+", " ", text).strip()


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """
    OpenAlex ne fournit pas l'abstract en clair mais un "inverted index"
    (mot -> positions). On le reconstruit en texte normal.
    """
    if not inverted_index:
        return None
    position_to_word = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_to_word[pos] = word
    if not position_to_word:
        return None
    max_pos = max(position_to_word.keys())
    words = [position_to_word.get(i, "") for i in range(max_pos + 1)]
    return clean_text(" ".join(words))


def classify_study_type(work: dict) -> str:
    """
    Heuristique : on regarde le type OpenAlex + des mots-clés dans le titre
    pour approximer un niveau de preuve (meta-analysis > rct > observational).
    """
    title_lower = (work.get("title") or "").lower()
    work_type = (work.get("type") or "").lower()

    if "meta-analysis" in title_lower or "systematic review" in title_lower or work_type == "review":
        return "meta-analysis"
    if "randomized" in title_lower or "randomised" in title_lower or "rct" in title_lower:
        return "rct"
    if any(kw in title_lower for kw in ["cohort", "observational", "cross-sectional", "survey"]):
        return "observational"
    return "unknown"


def fetch_works_for_query(query: str, per_page: int = RESULTS_PER_QUERY) -> list[dict]:
    params = {
        "search": query,
        "per_page": per_page,
        "filter": "has_abstract:true",
    }
    if MAILTO:
        params["mailto"] = MAILTO

    resp = requests.get(API_URL, params=params, timeout=30)
    if resp.status_code == 429:
        time.sleep(5)
        resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def upsert_paper(cur, work: dict, query: str):
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if not abstract:
        return  # on ignore les papiers sans abstract exploitable

    paper_id = work.get("id", "").replace("https://openalex.org/", "")
    if not paper_id:
        return

    study_type = classify_study_type(work)
    venue = None
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    if source:
        venue = source.get("display_name")

    cur.execute(
        """
        INSERT INTO papers (paper_id, title, abstract, year, venue,
                             study_type, citation_count, url, query_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_id) DO UPDATE SET
            citation_count = EXCLUDED.citation_count,
            fetched_at = now()
        """,
        (
            paper_id,
            clean_text(work.get("title")),
            abstract,
            work.get("publication_year"),
            venue,
            study_type,
            work.get("cited_by_count") or 0,
            primary_location.get("landing_page_url") or work.get("id"),
            query,
        ),
    )


def main():
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    total_ingested = 0
    for query in tqdm(QUERIES, desc="Requêtes OpenAlex"):
        try:
            works = fetch_works_for_query(query)
        except requests.HTTPError as e:
            print(f"[WARN] échec pour la requête '{query}': {e}")
            continue

        for work in works:
            upsert_paper(cur, work, query)
            total_ingested += 1

        time.sleep(0.2)  # largement suffisant, OpenAlex est généreux

    cur.close()
    conn.close()
    print(f"Ingestion terminée : {total_ingested} papiers traités.")


if __name__ == "__main__":
    main()
