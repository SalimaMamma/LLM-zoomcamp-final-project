"""
Charge les triplets (sujet, relation, objet) stockés en Postgres dans un
graphe NetworkX, et fournit des primitives de requêtage simples :
recherche par proximité d'entité (nom approximatif) + traversée pour
récupérer les relations directes et les contradictions.
"""
import os

import networkx as nx
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "scifit"),
        user=os.getenv("POSTGRES_USER", "scifit"),
        password=os.getenv("POSTGRES_PASSWORD", "scifit_local_pw"),
    )


def get_papers_metadata(paper_ids: list[str]) -> dict[str, dict]:
    """Récupère titre/année/type d'étude/URL pour une liste de paper_id --
    permet de retracer une relation du graphe (sujet/relation/objet, sans
    contexte en soi) jusqu'au papier source dont elle a été extraite. Voir
    streamlit_app/app.py, mode "graph"."""
    paper_ids = [p for p in dict.fromkeys(paper_ids) if p]  # dédoublonne, garde l'ordre
    if not paper_ids:
        return {}
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT paper_id, title, year, study_type, url FROM papers WHERE paper_id = ANY(%s)",
        (paper_ids,),
    )
    metadata = {
        row[0]: {"title": row[1], "year": row[2], "study_type": row[3], "url": row[4]}
        for row in cur.fetchall()
    }
    cur.close()
    conn.close()
    return metadata


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._load_from_postgres()

    def _load_from_postgres(self):
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT source_paper_id, subject, relation, object FROM graph_edges")
        for paper_id, subject, relation, obj in cur.fetchall():
            self.graph.add_edge(subject, obj, relation=relation, paper_id=paper_id)
        cur.close()
        conn.close()

    def find_matching_nodes(self, term: str) -> list[str]:
        """Recherche approximative : nœuds dont le nom contient le terme."""
        term_lower = term.lower()
        return [n for n in self.graph.nodes if term_lower in n or n in term_lower]

    def get_relations_for_entity(self, term: str, max_hops: int = 1) -> list[dict]:
        """Récupère les relations directes (et éventuellement à 2 sauts) autour d'une entité."""
        matches = self.find_matching_nodes(term)
        results = []
        seen_nodes = set(matches)
        frontier = set(matches)

        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                for _, target, data in self.graph.out_edges(node, data=True):
                    results.append(
                        {
                            "subject": node,
                            "relation": data["relation"],
                            "object": target,
                            "paper_id": data["paper_id"],
                        }
                    )
                    if target not in seen_nodes:
                        next_frontier.add(target)
                for source, _, data in self.graph.in_edges(node, data=True):
                    results.append(
                        {
                            "subject": source,
                            "relation": data["relation"],
                            "object": node,
                            "paper_id": data["paper_id"],
                        }
                    )
                    if source not in seen_nodes:
                        next_frontier.add(source)
            seen_nodes |= next_frontier
            frontier = next_frontier

        return results

    def find_contradictions(self, term: str) -> list[dict]:
        """Filtre spécifiquement les relations de type contradicts/confirms autour d'une entité."""
        relations = self.get_relations_for_entity(term, max_hops=1)
        return [r for r in relations if r["relation"] in ("contradicts", "confirms")]
