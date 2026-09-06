"""
Évaluation du retrieval sur le jeu de questions synthétiques
(synthetic_questions.json, généré par generate_synthetic_questions.py).

Contrairement à retrieval_eval.py (8 questions annotées à la main, "hit"
approximé par présence d'un mot-clé), chaque question ici est rattachée à
un gold_chunk_id ET un gold_paper_id exacts -- on peut donc calculer un
vrai Recall@k / MRR au niveau du chunk (le passage exact) et au niveau du
papier (la bonne source, même si un autre chunk du même papier a été
récupéré à la place -- une histoire de granularité du chunking, pas une
vraie erreur de retrieval).

Usage:
    python src/eval/retrieval_eval_synthetic.py
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "graphrag"))

from hybrid import HybridRetriever, rerank_chunks  # noqa: E402
from graph_store import KnowledgeGraph  # noqa: E402

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "synthetic_questions.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval_results")
TOP_K = 8


def rank_of(chunks: list[dict], key: str, gold_value: str) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.get(key) == gold_value:
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 1 / rank if rank is not None else 0.0


def score(chunks: list[dict], q: dict) -> dict:
    chunk_rank = rank_of(chunks, "chunk_id", q["gold_chunk_id"])
    paper_rank = rank_of(chunks, "paper_id", q["gold_paper_id"])
    return {
        "chunk_hit": 1 if chunk_rank is not None else 0,
        "chunk_rr": reciprocal_rank(chunk_rank),
        "paper_hit": 1 if paper_rank is not None else 0,
        "paper_rr": reciprocal_rank(paper_rank),
    }


def aggregate(label: str, rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "mode": label,
        "n_questions": n,
        f"chunk_recall@{TOP_K}": sum(r["chunk_hit"] for r in rows) / n,
        "chunk_mrr": sum(r["chunk_rr"] for r in rows) / n,
        f"paper_recall@{TOP_K}": sum(r["paper_hit"] for r in rows) / n,
        "paper_mrr": sum(r["paper_rr"] for r in rows) / n,
    }


def evaluate_mode(retriever: HybridRetriever, questions: list[dict], mode: str, rerank: bool) -> dict:
    rows = [score(retriever.search(q["question"], top_k=TOP_K, mode=mode, rerank=rerank), q) for q in questions]
    return aggregate(f"{mode}+rerank" if rerank else mode, rows)


def evaluate_graph(kg: KnowledgeGraph, questions: list[dict], rerank: bool) -> dict:
    rows = []
    for q in questions:
        # même appel que dans l'app (streamlit_app/app.py, mode "graph") :
        # la question brute sert de terme de recherche approximatif
        relations = kg.get_relations_for_entity(q["question"])
        relations = relations[: TOP_K * 3 if rerank else TOP_K]
        chunks = [
            {"content": f"{r['subject']} {r['relation']} {r['object']}", "paper_id": r.get("paper_id"), "chunk_id": None}
            for r in relations
        ]
        if rerank and chunks:
            chunks, _ = rerank_chunks(q["question"], chunks, top_k=TOP_K)
        rows.append(score(chunks, q))
    return aggregate("graph+rerank" if rerank else "graph", rows)


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    retriever = HybridRetriever()
    kg = KnowledgeGraph()

    results = []
    for mode in ("vector", "bm25", "hybrid"):
        results.append(evaluate_mode(retriever, questions, mode, rerank=False))
        results.append(evaluate_mode(retriever, questions, mode, rerank=True))
    results.append(evaluate_graph(kg, questions, rerank=False))
    results.append(evaluate_graph(kg, questions, rerank=True))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "retrieval_eval_synthetic.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_questions": len(questions), "results": results}, f, indent=2, ensure_ascii=False)

    print(f"Évaluation sur {len(questions)} questions synthétiques (gold chunk/paper exacts) :")
    print(f"{'mode':15s} {'chunk_recall@'+str(TOP_K):>15s} {'chunk_mrr':>10s} {'paper_recall@'+str(TOP_K):>15s} {'paper_mrr':>10s}")
    for r in results:
        print(
            f"{r['mode']:15s} {r[f'chunk_recall@{TOP_K}']:>15.2f} {r['chunk_mrr']:>10.2f} "
            f"{r[f'paper_recall@{TOP_K}']:>15.2f} {r['paper_mrr']:>10.2f}"
        )
    print(f"\nDétails sauvegardés dans {out_path}")


if __name__ == "__main__":
    main()
