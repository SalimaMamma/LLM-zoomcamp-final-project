# Monitoring

Retour au [README](../README.md).

## Comment le feedback est collecté

Chaque réponse affichée dans l'app Streamlit ([streamlit_app/app.py](../streamlit_app/app.py))
propose deux boutons **👍 Utile** / **👎 Pas utile**. Un clic insère une
ligne dans la table Postgres `feedback` :

```sql
INSERT INTO feedback (question, answer, rating, retrieval_mode, latency_ms)
VALUES (%s, %s, %s, %s, %s)
```

`rating` (+1/-1), `retrieval_mode` (hybrid/vector/bm25/graph) et
`latency_ms` (mesuré côté app, du clic sur "Vérifier" à la réponse générée)
sont enregistrés à chaque fois — pas seulement un compteur agrégé, ce qui
permet de creuser par mode ou dans le temps.

## Dashboard Grafana

Provisionné automatiquement au démarrage (`monitoring/provisioning/`,
aucune configuration manuelle dans l'UI Grafana requise) — disponible dès
`docker compose up -d` sur http://localhost:3001 (`admin`/`admin`),
dossier **SciFit-Check**.

![Dashboard Grafana](images/03_grafana_dashboard.png)

6 panels ([JSON source](../monitoring/provisioning/dashboards/scifit_overview.json)) :

| Panel | Type | Source | Ce qu'il montre |
|---|---|---|---|
| Volume de questions par jour | timeseries | `feedback` | Usage dans le temps |
| Taux de satisfaction (%) | stat | `feedback` | % de ratings positifs |
| Latence moyenne (ms) | stat | `feedback` | Temps de réponse moyen bout-en-bout |
| Répartition des modes de retrieval utilisés | piechart | `feedback` | hybrid vs vector vs bm25 vs graph |
| Feedback positif vs négatif par mode | barchart | `feedback` | Qualité perçue par mode de retrieval |
| Papiers indexés par niveau de preuve | piechart | `papers` | Composition du corpus (meta-analysis/rct/observational/unknown) |

Le dernier panel ne dépend pas de la table `feedback` — il reflète l'état du
corpus indexé et fonctionne dès l'ingestion, avant toute interaction
utilisateur. C'est celui visible sur la capture ci-dessus : **462 papiers
sur 561 (82%) sont classés `unknown`** — un vrai angle mort à noter
honnêtement plutôt qu'à masquer : `classify_study_type()`
([src/ingestion/europepmc.py](../src/ingestion/europepmc.py)) s'appuie sur
`pubTypeList`, disponible et fiable côté Europe PMC mais absent des
métadonnées OpenAlex, qui alimente une bonne partie du corpus — piste
d'amélioration : classifier `study_type` par heuristique sur le titre/abstract
pour les papiers OpenAlex, ou n'ingérer OpenAlex qu'en complément ciblé.

### Les 5 autres panels (basés sur `feedback`)

Au moment de la rédaction de cette doc, la table `feedback` est vide (0
interaction) et ces panels affichent "No data" — pas un bug, juste pas
encore d'usage réel. Deux façons de les peupler :

1. **Usage normal** : ouvrir l'app, poser des questions, cliquer 👍/👎.
2. **Script de démo** ([`src/eval/seed_feedback_demo.py`](../src/eval/seed_feedback_demo.py)) :
   fait tourner le pipeline réel (retrieval + génération Groq) sur les 8
   questions annotées × 4 modes, et enregistre chaque interaction dans
   `feedback` exactement comme le ferait l'app. Le rating n'est pas un avis
   humain simulé au hasard : il est dérivé d'une vérification automatique
   du format de réponse attendu (citation de source + ligne "niveau de
   preuve" présentes, cf. prompt dans `src/llm/answer.py`).
   ```bash
   docker compose exec app python src/eval/seed_feedback_demo.py
   ```
   ⚠️ Consomme ~32 appels Groq — voir
   [quota Groq](setup.md#quota-groq-gratuit) si le compte est déjà proche
   de sa limite journalière au moment de l'exécution.

Un bug de configuration a été corrigé sur les deux panels `piechart`
pendant la préparation de cette doc : sans `reduceOptions.values: true`
explicite, Grafana (v11) réduit toutes les lignes d'une requête table à une
seule valeur agrégée nommée "total" au lieu de tracer une tranche par
catégorie — visible immédiatement en testant le dashboard avec de vraies
données plutôt qu'en committant la config à l'aveugle.

## Ce qui manquerait pour aller plus loin

- **Alerting** : Grafana le permet nativement (onglet "Alert rules" déjà
  visible dans l'UI provisionnée) — pas configuré ici, mais un seuil sur le
  taux de satisfaction ou la latence serait la suite logique.
- **Logs applicatifs structurés** : actuellement seul le feedback explicite
  est tracé ; les erreurs (ex: retrieval vide, échec Groq) ne remontent
  qu'en `st.warning()` côté UI, pas dans une table consultable.
