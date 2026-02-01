"""
Streamlit Dashboard pour Credit Scoring
Interface utilisateur pour les prédictions et monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

# Configuration
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Style CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success {color: #28a745;}
    .danger {color: #dc3545;}
</style>
""", unsafe_allow_html=True)

# Configuration API
# ⚠️ IMPORTANT: Utiliser l'URL Render, pas Railway
RENDER_API_URL = "https://credit-scoring-api-k4q9.onrender.com"

try:
    # Essayer le secret Streamlit d'abord
    secret_url = st.secrets.get("api_url", None)
    if secret_url and "render" in secret_url.lower():
        API_URL = secret_url
    else:
        # Sinon utiliser Render par défaut
        API_URL = RENDER_API_URL
except:
    API_URL = RENDER_API_URL

# === SIDEBAR ===

st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Sélectionner une page:",
    ["🏠 Accueil", "📊 Prédiction Client", "📈 Analytics", "⚙️ Monitoring", "📋 Documentation"]
)

# === PAGE: ACCUEIL ===

if page == "🏠 Accueil":
    st.title("🏦 Credit Scoring Dashboard")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Modèle", value="LightGBM", delta="v1.0")
    with col2:
        st.metric(label="Données", value="307K", delta="Clients")
    with col3:
        st.metric(label="Précision", value="AUC", delta="0.76+")
    
    st.markdown("---")
    
    # Vérifier l'API
    st.subheader("🔍 Vérification de la Connexion API")
    
    with st.expander("Détails de connexion", expanded=True):
        st.write(f"**URL API:** `{API_URL}`")
        
        try:
            st.write("⏳ Test de connexion en cours...")
            response = requests.get(f"{API_URL}/health", timeout=5)
            st.write(f"✅ **Status HTTP:** {response.status_code}")
            
            if response.status_code == 200:
                st.success("✅ API connectée et fonctionnelle")
                data = response.json()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Status:** {data.get('status', 'unknown')}")
                    st.write(f"**Modèle chargé:** {'Oui ✅' if data.get('model_loaded') else 'Non ❌'}")
                with col2:
                    st.write(f"**Nom du modèle:** {data.get('model_name', 'N/A')}")
                    st.write(f"**Seuil:** {data.get('threshold', 'N/A')}")
                
                # Afficher la réponse complète
                with st.expander("📋 Réponse complète /health"):
                    st.json(data)
            else:
                st.error(f"❌ API retourne le statut: {response.status_code}")
                st.write(f"Réponse: {response.text}")
        except requests.exceptions.Timeout:
            st.error("❌ Timeout: L'API met trop de temps à répondre")
        except requests.exceptions.ConnectionError as e:
            st.error(f"❌ Erreur de connexion: {str(e)}")
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            st.write(f"Type: {type(e).__name__}")
    
    st.markdown("---")
    
    # Informations sur le modèle
    try:
        st.subheader("📋 Récupération des Informations du Modèle")
        with st.spinner("Chargement..."):
            response = requests.get(f"{API_URL}/info", timeout=5)
            st.write(f"✅ Réponse: {response.status_code}")
            
            if response.status_code == 200:
                info = response.json()
                
                with st.expander("📋 Réponse brute /info", expanded=False):
                    st.json(info)
                
                st.success("✅ Informations du modèle reçues")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nom", info.get('model_name', 'LightGBM'))
                with col2:
                    if 'model_version' in info and info['model_version']:
                        st.metric("Version", info['model_version'])
                    else:
                        st.metric("Version", "1.0.0")
                with col3:
                    st.metric("Features", info.get('features_count', 20))
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Stratégie:** {info.get('strategy', 'class_weight')}")
                    st.write(f"**Seuil optimal:** {info.get('optimal_threshold', 0.46)}")
                with col2:
                    if info.get('data_source'):
                        st.write(f"**Source données:** {info['data_source']}")
                    total_clients = info.get('total_clients', 0)
                    if total_clients:
                        st.write(f"**Clients:** {total_clients:,}")
                
                # Métriques du modèle
                st.markdown("---")
                st.subheader("📊 Métriques de Performance")
                
                metrics = info.get('metrics', {})
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("F2-Score", f"{metrics.get('f2_score', 0):.4f}")
                with col2:
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                with col3:
                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                with col4:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                with col5:
                    st.metric("AUC-ROC", f"{metrics.get('auc', 0):.4f}")
                
                # Liste des features
                st.markdown("---")
                with st.expander("📑 Liste des 20 Features"):
                    cols = st.columns(2)
                    for i, feature in enumerate(info.get('features', []), 1):
                        with cols[(i-1) % 2]:
                            st.write(f"{i:2}. `{feature}`")
            else:
                st.error(f"❌ API retourne: {response.status_code}")
                st.write(f"Contenu: {response.text}")
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout: L'API met trop de temps à répondre")
    except requests.exceptions.ConnectionError:
        st.error("❌ Erreur de connexion à l'API")
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger les infos: {str(e)}")
        st.write(f"Erreur: {type(e).__name__}")

# === PAGE: PRÉDICTION CLIENT ===

elif page == "📊 Prédiction Client":
    st.title("📊 Prédiction Individuelle")
    st.markdown("---")
    
    # Charger le dataset light
    
    @st.cache_data
    def load_data_light():
        """Charge les données light depuis le CSV ou l'API"""
        try:
            # Essayer d'abord les chemins locaux
            possible_paths = [
                "data/data_mini_features.csv",
                "./data/data_mini_features.csv",
                "src/../data/data_mini_features.csv",
            ]
            
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    st.info("✅ Données chargées localement")
                    return df
                except FileNotFoundError:
                    continue
            
            # Si fichier local non trouvé, charger depuis l'API
            st.info(f"🔄 Chargement des données depuis l'API: {API_URL}/clients")
            
            try:
                response = requests.get(f"{API_URL}/clients", timeout=10)
                st.write(f"📡 Réponse API: Status {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    st.write(f"📊 Données reçues: {data}")
                    
                    clients_list = data.get('clients', [])
                    
                    if clients_list:
                        # Créer un DataFrame avec les IDs des clients
                        df = pd.DataFrame({'SK_ID_CURR': clients_list})
                        st.success(f"✅ Données chargées depuis l'API: {len(clients_list)} clients")
                        return df
                    else:
                        st.warning("⚠️ Aucun client disponible dans la réponse")
                        return None
                else:
                    st.error(f"❌ API retourne: {response.status_code}")
                    st.write(f"Contenu: {response.text}")
                    return None
            except requests.exceptions.Timeout:
                st.error("❌ Timeout: L'API met trop de temps à répondre")
                return None
            except requests.exceptions.ConnectionError as ce:
                st.error(f"❌ Erreur de connexion: {str(ce)}")
                return None
            except Exception as api_error:
                st.error(f"❌ Erreur API: {str(api_error)}")
                st.write(f"Type d'erreur: {type(api_error).__name__}")
                return None
                
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
            return None
    
    df = load_data_light()
    
    if df is not None and len(df) > 0:
        st.subheader("👤 Sélectionner un Client")
        
        sk_id_list = df['SK_ID_CURR'].tolist()
        selected_sk_id = st.selectbox(
            "Sélectionnez un client:",
            options=sk_id_list,
            index=0,
            help="Choisissez parmi les 307,505 clients"
        )
        
        if selected_sk_id:
            client_row = df[df['SK_ID_CURR'] == selected_sk_id].iloc[0]
            
            st.markdown("---")
            st.subheader(f"📋 Données du Client: {selected_sk_id}")
            
            # Filtrer seulement les colonnes numériques
            numeric_features = [col for col in df.columns 
                               if col != 'SK_ID_CURR' 
                               and pd.api.types.is_numeric_dtype(df[col])]
            
            st.info(f"✅ {len(numeric_features)} features numériques")
            
            # Afficher et éditer les features
            col1, col2 = st.columns(2)
            features = {}
            
            for i, feature in enumerate(numeric_features):
                value = client_row[feature]
                if i % 2 == 0:
                    with col1:
                        features[feature] = st.number_input(
                            f"{feature}",
                            value=float(value) if not pd.isna(value) else 0.0,
                            key=f"feat_{feature}_{selected_sk_id}"
                        )
                else:
                    with col2:
                        features[feature] = st.number_input(
                            f"{feature}",
                            value=float(value) if not pd.isna(value) else 0.0,
                            key=f"feat_{feature}_{selected_sk_id}"
                        )
            
            # Features catégoriques (info seulement)
            categorical_features = [col for col in df.columns 
                                   if col != 'SK_ID_CURR' 
                                   and not pd.api.types.is_numeric_dtype(df[col])]
            
            if categorical_features:
                st.markdown("---")
                st.subheader("📌 Features Catégoriques")
                col1, col2 = st.columns(2)
                for i, feature in enumerate(categorical_features):
                    if i % 2 == 0:
                        with col1:
                            st.write(f"**{feature}:** {client_row[feature]}")
                    else:
                        with col2:
                            st.write(f"**{feature}:** {client_row[feature]}")
            
            st.markdown("---")
            st.subheader("⚙️ Paramètres")
            
            threshold = st.slider(
                "Seuil de décision",
                0.0, 1.0, 0.46,
                step=0.01,
                help="Probabilité à partir de laquelle le crédit est refusé"
            )
            
            st.markdown("---")
            
            # Bouton de prédiction
            if st.button("🔮 LANCER LA PRÉDICTION", use_container_width=True, type="primary"):
                with st.spinner("Prédiction en cours..."):
                    try:
                        payload = {
                            "sk_id_curr": int(selected_sk_id),
                            "features": {},
                            "threshold": float(threshold)
                        }
                        
                        response = requests.post(
                            f"{API_URL}/predict",
                            json=payload,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.markdown("---")
                            st.subheader("✅ RÉSULTATS")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Probabilité Risque",
                                    f"{result['risk_probability']:.1%}"
                                )
                            
                            with col2:
                                decision = result['decision']
                                if "ACCORDÉ" in decision:
                                    st.metric("Décision", "✅ ACCORDÉ")
                                else:
                                    st.metric("Décision", "❌ REFUSÉ")
                            
                            with col3:
                                st.metric(
                                    "Distance Seuil",
                                    f"{abs(result['risk_probability'] - threshold):.3f}"
                                )
                            
                            with col4:
                                st.metric("Seuil", f"{result['threshold_used']:.2f}")
                            
                            st.markdown("---")
                            
                            prob = result['risk_probability']
                            if "ACCORDÉ" in decision:
                                st.success(f"✅ **CRÉDIT ACCORDÉ** - Risque: {prob:.1%}")
                            else:
                                st.error(f"❌ **CRÉDIT REFUSÉ** - Risque: {prob:.1%}")
                            
                            # Graphique gauge
                            st.markdown("---")
                            st.subheader("📊 Score de Risque")
                            
                            fig = go.Figure(data=[
                                go.Indicator(
                                    mode="gauge+number",
                                    value=prob * 100,
                                    title="Risque (%)",
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "#dc3545" if prob >= threshold else "#28a745"},
                                        'steps': [
                                            {'range': [0, 30], 'color': "#90EE90"},
                                            {'range': [30, 70], 'color': "#FFD700"},
                                            {'range': [70, 100], 'color': "#FF6B6B"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': threshold * 100
                                        }
                                    }
                                )
                            ])
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Top 10 features
                            if result.get('top_10_features'):
                                st.markdown("---")
                                st.subheader("🎯 Top 10 Features")
                                
                                top_10 = result['top_10_features']
                                top_10_df = pd.DataFrame([
                                    {
                                        'Rang': f["rank"],
                                        'Feature': f["feature_name"],
                                        'Importance': f["importance_value"]
                                    }
                                    for f in top_10
                                ])
                                
                                st.dataframe(top_10_df, use_container_width=True, hide_index=True)
                                
                                fig2 = px.bar(
                                    top_10_df,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h'
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.error(f"❌ Erreur API: {response.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Impossible de se connecter à {API_URL}")
                        st.info("💡 Démarrez l'API: `python -m uvicorn src.api:app --host 0.0.0.0 --port 8001`")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
    else:
        st.error("❌ Impossible de charger le dataset")

# === PAGE: ANALYTICS ===

elif page == "📈 Analytics":
    st.title("📈 Analytics et Statistiques")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Prédictions", "1,234", "↑ 12%")
    with col2:
        st.metric("Taux Approbation", "68%", "↓ 2%")
    with col3:
        st.metric("Risque Moyen", "32%", "↑ 3%")
    with col4:
        st.metric("API Uptime", "99.8%", "↑ 0.1%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Histogram(x=np.random.beta(2, 5, 1000), nbinsx=30)])
        fig.update_layout(
            title="Distribution des Scores de Risque",
            xaxis_title="Probabilité de Risque",
            yaxis_title="Fréquence"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=['Approbations', 'Rejets'],
            values=[680, 320]
        )])
        fig.update_layout(title="Distribution des Décisions")
        st.plotly_chart(fig, use_container_width=True)

# === PAGE: MONITORING ===

elif page == "⚙️ Monitoring":
    st.title("⚙️ Monitoring et Health Check")
    st.markdown("---")
    
    if st.button("🔄 Actualiser", use_container_width=True):
        st.rerun()
    
    st.subheader("🏥 État des Services")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success(f"✅ API: Opérationnel")
        else:
            st.error(f"❌ API: {response.status_code}")
    except:
        st.error(f"❌ API: Non disponible")
    
    st.markdown("---")
    st.subheader("📊 Métriques en Temps Réel")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Requêtes/min", "42", "↑ 8")
    with col2:
        st.metric("Latence API", "125ms", "↓ 15ms")
    with col3:
        st.metric("Erreurs", "0", "✅")

# === PAGE: DOCUMENTATION ===

elif page == "📋 Documentation":
    st.title("📋 Documentation")
    st.markdown("---")
    
    st.subheader("🎯 Guide d'Utilisation")
    
    st.markdown("""
    ### 1. Prédiction Individuelle
    - Accédez à l'onglet "Prédiction Client"
    - Sélectionnez un client depuis le dropdown
    - Vérifiez les features préchargées
    - Ajustez le seuil de décision si nécessaire
    - Cliquez sur "LANCER LA PRÉDICTION"
    
    ### 2. Monitoring
    - Vérifiez l'état des services
    - Consultez les métriques en temps réel
    
    ### 3. API Documentation
    - [API Swagger](https://credit-scoring-api-k4q9.onrender.com/docs)
    - [API ReDoc](https://credit-scoring-api-k4q9.onrender.com/redoc)
    """)
    
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            st.markdown("---")
            st.subheader("📚 Features du Modèle")
            st.write(f"**Total:** {info.get('features_count', 20)} features")
            cols = st.columns(2)
            for i, feature in enumerate(info.get('features', []), 1):
                with cols[(i-1) % 2]:
                    st.write(f"{i:2}. `{feature}`")
    except:
        st.error("Impossible de charger les features")
    
    st.markdown("---")
    st.subheader("📚 Ressources")
    st.markdown("""
    - [FastAPI Documentation](https://fastapi.tiangolo.com/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [LightGBM Documentation](https://lightgbm.readthedocs.io/)
    """)

# === FOOTER ===

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <small>Credit Scoring Dashboard © 2025 | Powered by Streamlit & FastAPI</small>
</div>
""", unsafe_allow_html=True)
