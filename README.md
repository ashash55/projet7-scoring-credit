# 🏦 Credit Scoring Dashboard

Système de prédiction de risque de crédit utilisant **LightGBM** avec une interface moderne via **FastAPI + Streamlit**.

## 📋 Structure du Projet

```
projet7-scoring-credit/
├── src/                    # Code source principal
│   ├── api.py             # FastAPI (port 8001)
│   └── app.py             # Streamlit Dashboard (port 8501)
├── data/                  # Données
│   └── data_light_features.csv  # 307K clients
├── notebooks/             # Jupyter Notebooks & analyse EDA
│   ├── model.ipynb        # Notebook du modèle
│   └── data/              # Données originales (brutes)
├── models/                # Modèles entraînés (MLflow artifacts)
├── deployment/            # Docker + CI/CD + Guides
│   ├── Dockerfile         # Image Docker multi-stage
│   ├── docker-compose.yml # Orchestration locale
│   └── *.md               # Guides déploiement
├── requirements.txt       # Dépendances Python
└── README.md             # Ce fichier
```

## 🚀 Démarrage Rapide

### 1. Installation

```bash
# Cloner le repo
git clone https://github.com/ashash55/projet7-scoring-credit.git
cd projet7-scoring-credit

# Créer virtual env
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

### 2. Lancer les Services

**Terminal 1 - API (FastAPI sur port 8001):**
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 - Dashboard (Streamlit sur port 8501):**
```bash
streamlit run src/app.py --server.port 8501
```

Accès:
- 🏠 Dashboard: http://localhost:8501
- 📚 API Docs: http://localhost:8001/docs

## 📊 Modèle & Données

| Item | Détail |
|------|--------|
| **Modèle** | LightGBM avec `class_weight` |
| **Stratégie** | Optimisé pour F2-Score (rappel important) |
| **Seuil Optimal** | 0.46 |
| **Features** | 20 colonnes numériques/catégoriques |
| **Clients** | 307,505 dans `data_light_features.csv` |
| **AUC-ROC** | 0.7584 |
| **Recall** | 0.6143 |
| **Precision** | 0.1856 |

## 🔌 API Endpoints

### GET /health
Vérifier la santé de l'API
```bash
curl http://localhost:8001/health
```

### GET /clients
Liste des clients disponibles
```bash
curl http://localhost:8001/clients
```

### GET /info
Infos du modèle (features, seuil, métriques)
```bash
curl http://localhost:8001/info
```

### POST /predict
Prédiction pour un client
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"sk_id_curr": 100001, "features": {}, "threshold": 0.46}'
```

## 🎯 Fonctionnalités Streamlit

| Page | Description |
|------|-------------|
| **🏠 Accueil** | KPIs, infos modèle, métriques de performance |
| **📊 Prédiction** | Sélectionner client → éditer features → prédiction |
| **📈 Analytics** | Distributions, graphiques de décisions |
| **⚙️ Monitoring** | Health check, métriques en temps réel |
| **📋 Documentation** | Guide, features, ressources |

## 🐳 Docker & Deployment

### Local Docker Compose
```bash
cd deployment
docker-compose up
```

### Déployer sur Railway.app
Voir le guide complet dans `deployment/03_DEPLOYER_RAILWAY.md`

**Résumé:**
1. Push code sur GitHub
2. Créer compte Railway + connecter GitHub
3. Ajouter 4 secrets GitHub (Docker token, Railway token, etc.)
4. Railway déploie automatiquement via GitHub Actions

## 📁 Fichiers Clés

| Fichier | Rôle |
|---------|------|
| `src/api.py` | FastAPI - endpoints de prédiction |
| `src/app.py` | Streamlit - interface utilisateur |
| `data/data_light_features.csv` | Dataset clients + features |
| `notebooks/model.ipynb` | Entraînement et exploration du modèle |
| `requirements.txt` | Dépendances Python |
| `deployment/Dockerfile` | Image Docker production |
| `deployment/.github-workflows-deploy.yml` | CI/CD pipeline GitHub Actions |

## 🧪 Tests

```bash
# Lancer les tests
pytest deployment/test_api.py -v

# Avec couverture
pytest deployment/test_api.py --cov=src
```

## 📖 Documentation Supplémentaire

- **Guide Cloud**: `deployment/DEPLOYMENT_GUIDE.md`
- **Créer Comptes**: `deployment/01_CREER_COMPTES_CLOUD.md`
- **Ajouter Secrets**: `deployment/02_AJOUTER_SECRETS_GITHUB.md`
- **Déployer**: `deployment/03_DEPLOYER_RAILWAY.md`
- **Évaluation CE**: `deployment/CE_COMPLETION_GUIDE.md`

## 🌍 Déploiement Public

URL de démo (une fois déployée): `https://credit-scoring.railway.app`

Pour les collègues: **Pas d'installation nécessaire** - juste leur donner le lien!

## 👨‍💻 Technos

- **Backend**: FastAPI, uvicorn
- **Frontend**: Streamlit, Plotly
- **ML**: LightGBM, scikit-learn
- **Data**: pandas, numpy
- **Deployment**: Docker, GitHub Actions, Railway.app
- **Testing**: pytest

## 📝 Licence

Projet personnel - 2025

## 🤝 Support

Pour les problèmes:
1. Vérifier les logs: `api_logs.log`
2. Consulter l'API health: `http://localhost:8001/health`
3. Vérifier les données: `ls -la data/data_light_features.csv`
