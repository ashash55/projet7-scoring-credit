# 🚀 Déploiement Credit Scoring

Ce dossier contient tous les fichiers nécessaires pour déployer l'application sur le cloud.

## 📁 Fichiers

| Fichier | Utilité |
|---------|---------|
| `Dockerfile` | Image Docker multi-stage |
| `docker-compose.yml` | Orchestration locale (API + Streamlit) |
| `.github-workflows-deploy.yml` | CI/CD GitHub Actions |
| `test_api.py` | Tests unitaires (21 tests) |
| `requirements_prod.txt` | Dépendances production légères |
| `*.md` | Guides étape par étape |

## 🐳 Docker Local

### Tester avec Docker Compose

```bash
cd deployment
docker-compose up
```

Accès:
- API: http://localhost:8001
- Streamlit: http://localhost:8501

### Build Image Docker

```bash
cd deployment
docker build -t credit-scoring:latest .
docker run -p 8001:8001 -p 8501:8501 credit-scoring:latest
```

## 🧪 Tests

```bash
# Lancer les tests
pytest test_api.py -v

# Avec couverture
pytest test_api.py --cov=../src
```

## 📋 Guides de Déploiement

1. **01_CREER_COMPTES_CLOUD.md** - Créer comptes Docker Hub + Railway
2. **02_AJOUTER_SECRETS_GITHUB.md** - Ajouter secrets pour CI/CD  
3. **03_DEPLOYER_RAILWAY.md** - Déployer sur Railway.app

## 🌍 Déploiement Production

### Prérequis
- Code pushé sur GitHub
- Docker Hub account + token
- Railway.app account
- 4 secrets GitHub configurés

### Processus
1. Push code sur GitHub → déclenche CI/CD
2. GitHub Actions lance les tests
3. Si tests OK: build Docker image
4. Push image sur Docker Hub
5. Railway redéploie depuis Docker Hub

### Résultat Final
- URL accessible worldwide: `https://credit-scoring.railway.app`
- Auto-restart sur crash
- Logs accessibles via Railway dashboard
- Scaling automatique possible

## 📊 Architecture

```
GitHub (Code)
    ↓
GitHub Actions (Test + Build)
    ↓
Docker Hub (Image)
    ↓
Railway (Déploiement & Hosting)
    ↓
URL Public (Accès utilisateur)
```

## 🔧 Configuration

### Secrets GitHub Requis:
- `DOCKER_USERNAME` - Nom Docker Hub
- `DOCKER_PASSWORD` - Token Docker Hub
- `RAILWAY_TOKEN` - Token Railway  
- `RAILWAY_PROJECT_ID` - ID du projet Railway

### Variables d'Environnement:
- `API_PORT=8001`
- `STREAMLIT_PORT=8501`
- `PYTHONUNBUFFERED=1`

## ⚠️ Troubleshooting

### API ne démarre pas
```bash
docker logs <container_id>
# Vérifier: data/data_light_features.csv existe
```

### Streamlit refuse de se connecter à l'API
- Dans Railway: vérifier que les 2 services communiquent
- Vérifier `API_URL` dans `src/app.py`

### Tests échouent
```bash
# Démarrer API d'abord
python -m uvicorn src.api:app --host 0.0.0.0 --port 8001

# Dans une autre terminal
pytest deployment/test_api.py -v
```

## 📈 Monitoring

### Railway Dashboard
- https://railway.app/project/{PROJECT_ID}
- Voir logs en temps réel
- Gérer domains
- Voir consumption

### Logs API
```bash
# Localement
tail -f api_logs.log

# Railway
railway logs --service api
```

## 🚨 Alertes

Configuration recommandée dans Railway:
- Alert sur erreurs (status != 200)
- Alert sur latence API (> 2s)
- Alert sur CPU (> 80%)
- Alert sur mémoire (> 512MB)

## 💰 Coûts

Railway.app (estimé):
- **Gratuit**: 500h/mois
- **Payant**: $5/mth minimum
- CPU: $0.000278/hour
- RAM: $0.000694/hour
- Stockage: $0.10/GB/mth

## 🎓 Ressources

- [Railway Docs](https://docs.railway.app/)
- [Docker Docs](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Streamlit Cloud](https://docs.streamlit.io/streamlit-cloud/deploy-your-app)

---

**Besoin d'aide?** Consultez les fichiers `.md` dans ce dossier.
