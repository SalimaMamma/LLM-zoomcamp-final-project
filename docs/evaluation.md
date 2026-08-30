# Évaluation

Ce document détaille comment le retrieval et la génération sont évalués, avec
les résultats réels obtenus sur ce corpus. Retour au [README](../README.md).

## 1. Évaluation du retrieval

**Script :** [`src/eval/retrieval_eval.py`](../src/eval/retrieval_eval.py)
**Jeu de questions :** [`src/eval/questions_annotees.json`](../src/eval/questions_annotees.json)
— 8 questions en français, chacune annotée avec les mots-clés anglais
attendus dans une source pertinente (le corpus est en anglais, les questions
utilisateur sont en français).

### Méthodologie

On ne dispose pas de labels "ce chunk_id est pertinent pour cette question"
(ça demanderait une annotation manuelle chunk-par-chunk, hors scope solo).
À la place, on utilise un proxy simple mais objectif : un chunk est considéré
"pertinent" s'il contient au moins un des mots-clés attendus. Pour chaque
mode de retrieval, sur chaque question :

- **Recall@8** : 1 si un chunk pertinent apparaît dans les 8 premiers
  résultats, 0 sinon.
- **MRR** (Mean Reciprocal Rank) : `1 / rang` du premier chunk pertinent
  (0 si aucun).

Quatre modes sont comparés : `vector` (embeddings seuls), `bm25` (lexical
seul), `hybrid` (fusion RRF vecteur+BM25 — voir
[`src/retrieval/hybrid.py`](../src/retrieval/hybrid.py)), et `graph`
(traversée du graphe de connaissances extrait par LLM).

### Résultats (corpus actuel, 561 papiers / 1376 chunks)

| Mode     | Recall@8 | MRR    |
|----------|:--------:|:------:|
| vector   | 1.00     | 0.938  |
| bm25     | 0.875    | 0.625  |
| **hybrid** | **1.00** | **0.917** |
| graph    | 1.00     | 1.00   |

Régénérable avec :
```bash
docker compose exec app python src/eval/retrieval_eval.py
```
Résultats bruts : [`data/eval_results/retrieval_eval.json`](../data/eval_results/retrieval_eval.json).

### Pourquoi `hybrid` est le mode par défaut de l'app malgré ces chiffres

Sur cet échantillon de 8 questions, `vector` et `graph` affichent un MRR
nominalement supérieur à `hybrid`. Deux raisons pour ne pas en conclure
"il faut désactiver hybrid" :

1. **8 questions, c'est trop peu pour trancher statistiquement** — un
   écart de 0.02 sur 8 échantillons n'est pas significatif.
2. **Le proxy "graph" est structurellement avantagé par cette métrique** :
   `evaluate_graph()` compte un hit dès qu'il existe *une* relation pour le
   premier mot-clé de la question, sans notion de rang ni de pertinence du
   contenu retourné (contrairement à `vector`/`bm25`/`hybrid` qui classent
   réellement des chunks). Le MRR de 1.0 reflète donc "le terme existe dans
   le graphe", pas "la meilleure réponse est en position 1". Ce n'est pas
   comparable à la même métrique appliquée aux deux autres modes — limite du
   script d'éval, documentée ici plutôt que cachée.
3. **BM25 seul décroche nettement** (0.625 de MRR) sur des questions posées
   en français avec un corpus en anglais — l'appariement lexical exact rate
   les synonymes/traductions que les embeddings capturent. `hybrid` reste
   la valeur par défaut parce qu'il ne perd jamais beaucoup face au meilleur
   mode individuel tout en évitant le pire cas (une question purement
   terminologique où BM25 est fort, ou au contraire une question aux
   synonymes multiples où le vecteur est fort) — c'est un choix de
   robustesse, pas seulement de score brut sur cet échantillon.

## 2. Évaluation de la génération (LLM-as-judge)

**Script :** [`src/eval/llm_eval.py`](../src/eval/llm_eval.py)

Compare deux stratégies de prompt pour `generate_answer()` :

- `zero_shot` : réponse libre avec citations `[source: n]`.
- `structured_evidence` : réponse structurée en 3 parties (résumé / preuves
  citées / limites-contradictions).

Pour chaque question × stratégie, un second appel LLM ("juge", même modèle
Groq, température 0, sortie JSON forcée) note la **fidélité** (0-10) de la
réponse aux extraits fournis — c'est-à-dire si chaque affirmation est
traçable à une source, sans hallucination.

Régénérable avec :
```bash
docker compose exec app python src/eval/llm_eval.py
```
Résultats bruts : `data/eval_results/llm_eval.json`.

### Statut actuel des résultats

⚠️ **En attente d'exécution** : le compte Groq utilisé pour ce projet a
atteint son quota gratuit journalier (200k tokens/jour) pendant la
préparation de cette documentation, avant que cette évaluation ait pu
tourner jusqu'au bout. Le script est fonctionnel (voir le fix ci-dessous)
et sauvegarde ses résultats de façon incrémentale — relancer la commande
ci-dessus une fois le quota reconstitué (le lendemain sur le tier gratuit,
ou immédiatement avec un tier payant) suffit à obtenir les scores réels ;
ce fichier sera mis à jour avec le tableau de résultats à ce moment-là.

**Bug corrigé au passage** : le mode JSON forcé de Groq échoue parfois à
produire un JSON valide (guillemets typographiques non échappés dans la
justification) et levait une `BadRequestError` non interceptée, qui faisait
planter tout le script et perdre les résultats déjà obtenus. `llm_eval.py`
attrape maintenant ces erreurs par question (comme le fait déjà
`extract_entities.py` pour l'extraction du graphe) et sauvegarde après
chaque question plutôt qu'à la toute fin.

## 3. Ce que ces évaluations valident dans le pipeline

- Le retrieval hybride fonctionne bout-en-bout (Qdrant + BM25 + RRF) et
  n'est pas moins bon que ses composants pris séparément.
- Le prompt de génération force bien la citation de sources — condition
  nécessaire (mais pas suffisante, d'où le juge LLM) pour limiter
  l'hallucination.
- Les deux scripts d'évaluation sont automatisés et rejouables en une
  commande — pas une évaluation manuelle ponctuelle.
