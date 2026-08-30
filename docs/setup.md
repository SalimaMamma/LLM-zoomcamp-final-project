# Installation détaillée

Retour au [README](../README.md). Ce document couvre l'installation
complète, chaque variable d'environnement, et les problèmes réellement
rencontrés en développant ce projet (pas des problèmes théoriques).

## Prérequis

- Docker + Docker Compose (v2, `docker compose`, pas `docker-compose`)
- Une clé API Groq gratuite : https://console.groq.com (voir
  [limite de quota](#quota-groq-gratuit) plus bas)
- ~2 Go d'espace disque (image Docker + modèle d'embeddings local
  `bge-small-en-v1.5`, téléchargé au premier `build_index.py`)

Rien d'autre n'est requis sur la machine hôte : Postgres, Qdrant, Grafana et
l'app tournent tous en conteneurs.

## Étapes

```bash
git clone <url-de-ce-repo>
cd sci-fitness-rag

cp .env.example .env
# éditer .env : renseigner GROQ_API_KEY au minimum

docker compose up -d          # Postgres, Qdrant, Grafana, app (Streamlit)
```

Puis, dans l'ordre (chaque étape dépend de la précédente) :

```bash
# 1. Ingestion des papiers scientifiques (Europe PMC, biomédical strict)
docker compose exec app python src/ingestion/europepmc.py
# Alternative/complément : couverture plus large toutes disciplines
docker compose exec app python src/ingestion/openalex.py

# 2. Chunking + index vectoriel (Qdrant) + index BM25
docker compose exec app python src/retrieval/build_index.py

# 3. Extraction du graphe de connaissances (consomme du quota Groq)
docker compose exec app python src/graphrag/extract_entities.py
```

L'app est déjà démarrée par `docker compose up -d` (elle tourne en continu,
`streamlit run` est la commande par défaut du conteneur `app`) :

- **App** : http://localhost:8502
- **Grafana** : http://localhost:3001 (login `admin` / `admin`)

## Variables d'environnement

Toutes définies dans `.env` (voir `.env.example` pour les valeurs par défaut) :

| Variable | Obligatoire | Description |
|---|---|---|
| `GROQ_API_KEY` | **Oui** | Clé API Groq (génération de réponse + extraction de graphe). Gratuite sur console.groq.com. |
| `GROQ_MODEL` | Non | Modèle Groq à utiliser (défaut : `llama-3.3-70b-versatile`). |
| `OPENALEX_MAILTO` | Non | Ton email, pour rejoindre le "polite pool" OpenAlex (rate limit bien plus généreux). Aucune clé requise. |
| `SEMANTIC_SCHOLAR_API_KEY` | Non | Optionnelle, source d'ingestion alternative déconseillée (rate limit très strict sans clé). |
| `POSTGRES_HOST` / `PORT` / `DB` / `USER` / `PASSWORD` | Non | Défauts déjà cohérents avec `docker-compose.yml`. À ne changer que si tu modifies le compose. |
| `QDRANT_HOST` / `PORT` / `COLLECTION` | Non | Idem. |
| `EMBEDDING_MODEL` | Non | Modèle `sentence-transformers` local pour les embeddings (défaut : `BAAI/bge-small-en-v1.5`, CPU, gratuit). |

À l'intérieur des conteneurs, `docker-compose.yml` surcharge
`POSTGRES_HOST=postgres` et `QDRANT_HOST=qdrant` (résolution DNS interne à
Docker) — les valeurs `localhost` de `.env` sont pour un usage hors
conteneur (ex: lancer un script directement depuis ta machine avec un venv
Python, auquel cas les ports mappés sur l'hôte les rendent joignables).

## Ports exposés

| Service | Port hôte | Notes |
|---|---|---|
| App (Streamlit) | `8502` | Remappé depuis le défaut Streamlit 8501 — voir [pourquoi](#collision-de-ports) |
| Grafana | `3001` | Remappé depuis le défaut Grafana 3000 — même raison |
| Postgres | `5432` | Défaut |
| Qdrant | `6333` (REST), `6334` (gRPC) | Défaut |

## Dépannage

### Collision de ports

Si `docker compose up` échoue avec `port is already allocated`, un autre
projet sur ta machine utilise déjà ce port (fréquent avec Streamlit sur
8501 et Grafana sur 3000, deux ports par défaut très communs). Repère le
coupable :

```bash
docker ps --filter "publish=8501" --format "{{.Names}}"
```

Ce repo utilise déjà des ports remappés (8502/3001, voir `docker-compose.yml`
service `app`/`grafana`, section `ports`) pour éviter ça par défaut — mais
si TES autres projets utilisent 8502/3001, adapte-les à nouveau.

Un cas particulier : après un redémarrage de Docker Desktop, si `app` ou
`grafana` sortent en `Exited (255)` sans erreur explicite dans
`docker compose logs`, c'est presque toujours ce même problème de port
gagné par un autre conteneur qui a redémarré plus vite — vérifier avec
`docker compose ps -a` et `docker ps -a --filter publish=<port>`.

### Quota Groq gratuit

Le tier gratuit Groq limite à **200 000 tokens/jour** (tous modèles/appels
confondus sur le compte). L'extraction de graphe (`extract_entities.py`,
~1-2 appels par papier) et `llm_eval.py` (génération + jugement, 2 appels
par question × stratégie) sont les plus gourmands. Si tu vois :

```
groq.RateLimitError: Error code: 429 ... rate_limit_exceeded
```

le quota du jour est épuisé — attends le lendemain (reset quotidien) ou
passe sur un tier payant Groq. Ce n'est pas un bug du projet : les scripts
concernés (`llm_eval.py`, `seed_feedback_demo.py`) sauvegardent leurs
résultats de façon incrémentale, donc relancer la même commande le
lendemain reprend proprement plutôt que de tout refaire.

### Modifs locales non prises en compte par le conteneur

`docker-compose.yml` monte `./src`, `./streamlit_app` et `./data` en
volumes sur le service `app` — un fichier modifié localement est visible
immédiatement dans le conteneur, pas besoin de rebuild pour du code Python.
Un rebuild (`docker compose build app`) n'est nécessaire que si tu changes
`requirements.txt` ou le `Dockerfile` lui-même.

### `service "app" is not running`

Vérifie l'état réel avec `docker compose ps -a` (pas juste `ps`, qui masque
les conteneurs arrêtés) et regarde `docker compose logs app --tail 50` pour
la cause. Voir aussi [Collision de ports](#collision-de-ports) ci-dessus,
cause la plus fréquente rencontrée pendant le développement de ce projet.

## Reproductibilité

- Toutes les versions de dépendances sont épinglées dans `requirements.txt`.
- Le corpus n'est pas un fichier statique versionné : il est reconstruit à
  la demande via les scripts d'ingestion (APIs publiques, sans clé
  obligatoire pour Europe PMC/OpenAlex) — donc reproductible par n'importe
  qui, mais le contenu exact peut légèrement varier dans le temps (nouveaux
  papiers publiés/indexés).
- `data/eval_results/*.json` (résultats d'évaluation réels) sont versionnés
  dans le repo pour que les reviewers voient les scores sans tout relancer.
- `data/*.pkl` (index BM25) n'est pas versionné : régénérable en une
  commande (`build_index.py`), et le committer gonflerait le repo pour rien.
