"""
Ingestion incrémentale de papiers scientifiques via l'API Europe PMC.

Europe PMC agrège PubMed, PubMed Central et d'autres sources biomédicales.
Gratuit, sans clé API, pas de rate limit strict pour un usage raisonnable.
C'est l'alternative à privilégier si tu veux rester focus littérature
biomédicale/clinique plutôt que la couverture plus large d'OpenAlex.

Doc: https://europepmc.org/RestfulWebService

Usage:
    python src/ingestion/europepmc.py
"""
import os
import re
import time

import psycopg2
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

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


def classify_study_type(result: dict) -> str:
    """
    Europe PMC fournit parfois `pubType` (liste de types de publication).
    On s'appuie dessus, avec repli sur le titre.
    """
    pub_types = [t.lower() for t in result.get("pubTypeList", {}).get("pubType", [])]
    title_lower = (result.get("title") or "").lower()

    if any("meta-analysis" in t or "systematic review" in t for t in pub_types):
        return "meta-analysis"
    if any("randomized controlled trial" in t or "clinical trial" in t for t in pub_types):
        return "rct"
    if any("observational" in t or "comparative study" in t for t in pub_types):
        return "observational"

    # repli heuristique sur le titre si pubType absent/peu informatif
    if "meta-analysis" in title_lower or "systematic review" in title_lower:
        return "meta-analysis"
    if "randomized" in title_lower or "randomised" in title_lower:
        return "rct"
    return "unknown"


def fetch_results_for_query(query: str, page_size: int = RESULTS_PER_QUERY) -> list[dict]:
    params = {
        "query": f"{query} AND HAS_ABSTRACT:y",
        "format": "json",
        "pageSize": page_size,
        "resultType": "core",  # renvoie abstract + pubTypeList
    }
    resp = requests.get(API_URL, params=params, timeout=30)
    if resp.status_code == 429:
        time.sleep(5)
        resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("resultList", {}).get("result", [])


def upsert_paper(cur, result: dict, query: str) -> str:
    """Insère ou met à jour un papier. Retourne "inserted", "updated" ou "skipped"."""
    abstract = clean_text(result.get("abstractText"))
    if not abstract:
        return "skipped"

    # id PMID prioritaire, sinon PMCID, sinon DOI, sinon source+id
    paper_id = (
        result.get("pmid")
        or result.get("pmcid")
        or result.get("doi")
        or f"{result.get('source', 'unknown')}_{result.get('id', '')}"
    )
    if not paper_id:
        return "skipped"

    study_type = classify_study_type(result)
    year = None
    if result.get("pubYear"):
        try:
            year = int(result["pubYear"])
        except ValueError:
            year = None

    url = f"https://europepmc.org/article/{result.get('source', 'MED')}/{result.get('id', '')}"

    cur.execute(
        """
        INSERT INTO papers (paper_id, title, abstract, year, venue,
                             study_type, citation_count, url, query_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_id) DO UPDATE SET
            citation_count = EXCLUDED.citation_count,
            fetched_at = now()
        RETURNING (xmax = 0) AS was_insert
        """,
        (
            str(paper_id),
            clean_text(result.get("title")),
            abstract,
            year,
            result.get("journalTitle"),
            study_type,
            int(result.get("citedByCount") or 0),
            url,
            query,
        ),
    )
    was_insert = cur.fetchone()[0]
    return "inserted" if was_insert else "updated"


def main():
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    counts = {"inserted": 0, "updated": 0, "skipped": 0}
    for query in tqdm(QUERIES, desc="Requêtes Europe PMC"):
        try:
            results = fetch_results_for_query(query)
        except requests.HTTPError as e:
            print(f"[WARN] échec pour la requête '{query}': {e}")
            continue

        for result in results:
            status = upsert_paper(cur, result, query)
            counts[status] += 1

        time.sleep(0.3)

    cur.close()
    conn.close()
    total = sum(counts.values())
    print(
        f"Ingestion terminée : {total} résultats traités "
        f"({counts['inserted']} nouveaux, {counts['updated']} mis à jour, "
        f"{counts['skipped']} ignorés sans abstract/id)."
    )


if __name__ == "__main__":
    main()
