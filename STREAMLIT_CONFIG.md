# 🚀 Configuration Streamlit Cloud

## ⚙️ Configuration des Secrets

Pour que Streamlit Cloud utilise la bonne API Render, vous devez configurer les secrets:

### Étapes:

1. Allez sur: https://share.streamlit.io/admin/
2. Sélectionnez votre app: `projet7-scoring-credit`
3. Cliquez sur **Settings** ⚙️
4. Allez à l'onglet **Secrets**
5. Collez le contenu suivant:

```toml
api_url = "https://credit-scoring-api-k4q9.onrender.com"
```

6. Cliquez **Save**
7. L'app redémarrera automatiquement

### ✅ Vérification

Après configuration:
- Allez sur: https://projet7-scoring-credit-d9gw9jh9ancskwne9meijn.streamlit.app/
- Page "Accueil" → "Vérification de la Connexion API"
- Vous devriez voir ✅ Status HTTP: 200
- Les informations du modèle doivent s'afficher

### 🔗 URLs:

- **API Render:** https://credit-scoring-api-k4q9.onrender.com
  - Endpoints: `/health`, `/clients`, `/info`, `/predict`
  
- **Streamlit:** https://projet7-scoring-credit-d9gw9jh9ancskwne9meijn.streamlit.app/
  - Pages: Accueil, Prédiction, Analytics, Monitoring, Docs

### ⚠️ Troubleshooting

Si vous voyez toujours 404:

1. **Vérifiez l'URL** dans les secrets
   - Doit être: `https://credit-scoring-api-k4q9.onrender.com`
   - PAS: `https://projet7-scoring-credit-production.up.railway.app`

2. **Force refresh:**
   - Allez sur l'app Streamlit
   - Appuyez sur **R** pour forcer le reload
   - Ou videz le cache du navigateur

3. **Vérifiez l'API Render:**
   - https://credit-scoring-api-k4q9.onrender.com/health
   - Devrait retourner un JSON avec `"status": "healthy"`

4. **Logs:**
   - Streamlit Cloud: https://share.streamlit.io/admin/
   - Sélectionnez l'app → Logs
   - Render: Dashboard → Logs de votre service
