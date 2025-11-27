# 🚀 Guide Complet: Créer Comptes + Choisir Cloud

## 1️⃣ CRÉER UN COMPTE DOCKER HUB (Gratuit - 5 min)

### **Étape 1: Inscription**

1. Aller sur: https://hub.docker.com/signup
2. Remplir:
   ```
   Email: votre-email@gmail.com
   Username: ashash  (ou votre pseudo)
   Password: mot de passe fort
   ```
3. Cliquer: **Sign up**
4. Vérifier votre email (confirmation)

### **Étape 2: Créer un Access Token**

1. Login sur Docker Hub: https://hub.docker.com/
2. Cliquer sur votre profil (coin haut-droit)
3. Aller: **Account Settings**
4. Cliquer: **Security** (dans le menu gauche)
5. Cliquer: **New Access Token**
   ```
   Token name: github-actions
   Access permissions: Read & Write
   ```
6. Cliquer: **Generate**
7. **COPIER le token** (s'affiche UNE FOIS!)
   ```
   dckr_pat_xxxxxxxxxxxxxxxxxxxxx
   ```

### **✅ Vous avez:**
- ✓ DOCKER_USERNAME = `ashash`
- ✓ DOCKER_PASSWORD = `dckr_pat_xxxxxxxxxxxxxxxxxxxxx` 

--- 

## 2️⃣ CRÉER UN COMPTE RAILWAY (Gratuit - 5 min)

### **Étape 1: Inscription**

1. Aller sur: https://railway.app/
2. Cliquer: **Start Free** (ou Sign up)
3. Options:
   - Email + password
   - GitHub (plus facile!)
4. Cliquer: **Authorize railway-app** (si GitHub)

### **Étape 2: Créer un Access Token**

1. Login sur Railway: https://railway.app/
2. Cliquer: Settings (gear icon en haut-droit)
3. Aller: **Tokens**
4. Cliquer: **Create Token**
5. Donner un nom: `github-actions`
6. Cliquer: **Create**
7. **COPIER le token**
   ```
   rw_xxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### **Étape 3: Créer un Project**

1. Aller: https://railway.app/new
2. Cliquer: **Create New**
3. Sélectionner: **GitHub Repo**
4. Connecter votre repo
5. Railway crée automatiquement un project
6. Aller: **Settings** (onglet)
7. Aller: **General**
8. Copier: **Project ID**
   ```
   xxxxxxxxxxxxxxxxxxxxx
   ```

### **✅ Vous avez:**
- ✓ RAILWAY_TOKEN = `rw_xxxxxxxxxxxxxxxxxxxxxxxxxx`
- ✓ RAILWAY_PROJECT_ID = `xxxxxxxxxxxxxxxxxxxxx`

---

## 🎯 RÉSUMÉ: Les 4 Valeurs

| Secret | Valeur | Compte |
|--------|--------|--------|
| **DOCKER_USERNAME** | `ashash` | Docker Hub |
| **DOCKER_PASSWORD** | `dckr_pat_xxxx` | Docker Hub Token |
| **RAILWAY_TOKEN** | `rw_xxxx` | Railway Token |
| **RAILWAY_PROJECT_ID** | `xxxxxx` | Railway Project |

---

## 3️⃣ QUEL CLOUD CHOISIR?

### **OPTION 1: Railway (⭐ RECOMMANDÉ POUR VOUS)**

**Avantages:**
- ✅ Très simple à utiliser
- ✅ $5/mois (gratuit les premiers 500 heures)
- ✅ Déploie automatiquement depuis GitHub
- ✅ Support français possible
- ✅ Parfait pour démo + production légère

**Inconvénients:**
- ❌ Moins de features que AWS
- ❌ Limite de resources

**Coût:**
- Gratuit: 500 heures/mois = ~20 jours continu
- Payant: $5/mois minimum

**Pour vous:** ✅ **IDÉAL**

---

### **OPTION 2: Hugging Face Spaces (Gratuit)**

**Avantages:**
- ✅ Complètement gratuit
- ✅ Zéro configuration
- ✅ Déploie Streamlit en 1 clic
- ✅ Parfait pour démo

**Inconvénients:**
- ❌ Pas de CI/CD avancée
- ❌ Pas d'API séparée
- ❌ Limité en resources

**Coût:** Gratuit toujours

**Pour vous:** ✅ **Si vous voulez juste démo**

---

### **OPTION 3: AWS (Production réelle)**

**Avantages:**
- ✅ Le meilleur (mais complexe)
- ✅ Scalabilité illimitée
- ✅ Gratuit 1 an (AWS free tier)

**Inconvénients:**
- ❌ Complexe à mettre en place
- ❌ Cher après free tier (~$50/mois)
- ❌ Beaucoup de configuration

**Coût:**
- Gratuit 1 an (free tier)
- Puis: $20-100/mois selon usage

**Pour vous:** ❌ **Overkill pour une démo**

---

### **OPTION 4: Render.com**

**Avantages:**
- ✅ Simple
- ✅ Moins cher que AWS
- ✅ Déploie depuis GitHub

**Inconvénients:**
- ❌ Moins connu que Railway
- ❌ $7/mois (pas gratuit)

**Coût:** $7/mois

**Pour vous:** ❌ **Railway est mieux**

---

## 🎯 MON CHOIX POUR VOUS: RAILWAY ⭐

### **Pourquoi?**

1. **Simple** → Just push to GitHub = auto deploy
2. **Pas cher** → $5/mois (ou gratuit pendant 500h)
3. **Parfait pour démo** → Exactement ce qu'il vous faut
4. **Support CE3** → Satisfait le critère "déploiement cloud continu"

### **Coût Total:**
- Docker Hub: Gratuit (limite 2GB)
- Railway: $5/mois
- **Total: $5/mois** ✅

---

## 📋 RÉSUMÉ: Les 4 Secrets à Ajouter

### **Via Docker Hub:**
1. **DOCKER_USERNAME** = `ashash`
2. **DOCKER_PASSWORD** = Token copié de https://hub.docker.com/settings/security

### **Via Railway:**
3. **RAILWAY_TOKEN** = Token copié de https://railway.app/account/tokens
4. **RAILWAY_PROJECT_ID** = ID copié de Railway project settings

---

## ✅ CHECKLIST: Avant de Continuer

- [ ] Compte Docker Hub créé: https://hub.docker.com/
- [ ] Token Docker Hub généré et copié
- [ ] Compte Railway créé: https://railway.app/
- [ ] Token Railway généré et copié
- [ ] Project Railway créé et ID copié
- [ ] Prêt à ajouter les 4 secrets sur GitHub

---

## 🚀 PROCHAINE ÉTAPE

Une fois les comptes créés et les 4 valeurs copiées:

1. Aller sur GitHub: https://github.com/YOUR-USERNAME/YOUR-REPO/settings/secrets/actions
2. Ajouter les 4 secrets (voir guide suivant)
3. Push le code
4. Railway déploie automatiquement!

---

## 💡 IMPORTANT

- ✅ **DOCKER_PASSWORD** et **RAILWAY_TOKEN** = **SECRETS**
- ❌ Ne jamais les partager
- ❌ Ne jamais les mettre dans le code
- ✅ Les garder sauf sur GitHub Secrets

---

**Prêt?** Créez les comptes et revenez nous dire! 🎉
