"""
Compare vecteur seul / BM25 seul / hybride (RRF) / graphe sur le set de
questions annotées, avec les métriques Recall@k et MRR (approximées via
un matching de mots-clés attendus dans les chunks récupérés, faute de
gold labels chunk-par-chunk).

Usage:
    python src/eval/retrieval_eval.py
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "graphrag"))

from hybrid import HybridRetriever  # noqa: E402
from graph_store import KnowledgeGraph  # noqa: E402

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions_annotees.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval_results")
TOP_K = 8


def keyword_hit_rank(chunks: list[dict], keywords: list[str]) -> int | None:
    """Retourne le rang (1-indexé) du premier chunk contenant un des mots-clés attendus."""
    for rank, chunk in enumerate(chunks, start=1):
        content = chunk.get("content", "").lower()
        if any(kw.lower() in content for kw in keywords):
            return rank
    return None


def recall_at_k(rank: int | None) -> int:
    return 1 if rank is not None else 0


def reciprocal_rank(rank: int | None) -> float:
    return 1 / rank if rank is not None else 0.0


def evaluate_mode(retriever: HybridRetriever, questions: list[dict], mode: str) -> dict:
    recalls, rrs = [], []
    for q in questions:
        chunks = retriever.search(q["question"], top_k=TOP_K, mode=mode)
        rank = keyword_hit_rank(chunks, q["expected_keywords"])
        recalls.append(recall_at_k(rank))
        rrs.append(reciprocal_rank(rank))
    return {
        "mode": mode,
        f"recall@{TOP_K}": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
    }


def evaluate_graph(kg: KnowledgeGraph, questions: list[dict]) -> dict:
    recalls, rrs = [], []
    for q in questions:
        # heuristique simple : on cherche une entité correspondant au premier mot-clé
        primary_term = q["expected_keywords"][0]
        relations = kg.get_relations_for_entity(primary_term)
        hit = len(relations) > 0
        recalls.append(1 if hit else 0)
        rrs.append(1.0 if hit else 0.0)  # pas de notion de rang pour le graphe
    return {
        "mode": "graph",
        f"recall@{TOP_K}": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
    }


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    retriever = HybridRetriever()
    kg = KnowledgeGraph()

    results = [
        evaluate_mode(retriever, questions, "vector"),
        evaluate_mode(retriever, questions, "bm25"),
        evaluate_mode(retriever, questions, "hybrid"),
        evaluate_graph(kg, questions),
    ]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "retrieval_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Résultats de l'évaluation retrieval :")
    for r in results:
        print(f"  {r['mode']:8s} — recall@{TOP_K}: {r[f'recall@{TOP_K}']:.2f}  MRR: {r['mrr']:.2f}")
    print(f"\nDétails sauvegardés dans {out_path}")


if __name__ == "__main__":
    main()
