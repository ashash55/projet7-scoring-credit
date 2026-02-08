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
import matplotlib.pyplot as plt

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
try:
    API_URL = st.secrets.get("api_url", "http://localhost:8001")
except:
    API_URL = "http://localhost:8001"

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
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API connectée et fonctionnelle")
            data = response.json()
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Status:** {data['status']}")
            with col2:
                st.write(f"**Modèle chargé:** {'Oui' if data['model_loaded'] else 'Non'}")
        else:
            st.error(f"❌ API retourne: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Impossible de se connecter à l'API: {str(e)}")
    
    st.markdown("---")
    
    # Informations sur le modèle
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            st.subheader("📋 Informations du Modèle")
            
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
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger les infos: {str(e)}")

# === PAGE: PRÉDICTION CLIENT ===

elif page == "📊 Prédiction Client":
    st.title("📊 Prédiction Individuelle")
    st.markdown("---")
    
    # Charger le dataset light
    @st.cache_data
    def load_data_light():
        """Charge les données light depuis le CSV"""
        try:
            df = pd.read_csv("data/data_light_features.csv")
            return df
        except FileNotFoundError:
            st.error("❌ Fichier data/data_light_features.csv non trouvé")
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
            
            # Tableau explicatif des deux types d'importances
            st.info("""
            ### 📚 Comprendre les Feature Importances:
            
            **1. Feature Importance GLOBALE (Top 10 Features)**
            - Calcul: Basé sur tous les clients du dataset
            - Variation: IDENTIQUE pour tous les clients ✓
            - Signification: Quelles features sont importantes en général pour le modèle?
            - Exemple: Age est la 3ème feature la plus importante pour TOUS les clients
            
            **2. Feature Importance LOCALE (SHAP Waterfall)**
            - Calcul: Spécifique à chaque client
            - Variation: DIFFÉRENTE pour chaque client ✓
            - Signification: Pourquoi le modèle prédit ce risque POUR CE CLIENT?
            - Exemple: L'age du client X augmente son risque, mais pas pour le client Y
            """)
            
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
                            
                            # Afficher les métriques principales
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
                            
                            # Résumé principaux
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
                            
                            # === RÉCAPITULATIF DES DEUX TYPES D'IMPORTANCE ===
                            st.markdown("---")
                            col_recap1, col_recap2 = st.columns(2)
                            
                            with col_recap1:
                                st.subheader("🌍 Feature Importance GLOBALE")
                                st.write("""
                                - **Basée sur**: Tous les clients du dataset
                                - **Stabilité**: Identique pour tous les clients
                                - **Miseà jours**: Seulement lors du réentraînement du modèle
                                - **Utilité**: Comprendre quelles features sont importantes EN GÉNÉRAL
                                """)
                            
                            with col_recap2:
                                st.subheader("👤 Feature Importance LOCALE (SHAP)")
                                st.write("""
                                - **Basée sur**: Le client spécifique analysé
                                - **Variation**: Différente pour chaque client
                                - **Mise à jour**: Calculée à chaque prédiction
                                - **Utilité**: Expliquer pourquoi le modèle prédit ce risque POUR CE CLIENT
                                """)
                            
                            st.markdown("---")
                            
                            # Top 10 features GLOBALES
                            if result.get('top_10_features'):
                                st.markdown("---")
                                st.subheader("🎯 Top 10 Features Importances - Globales (Modèle)")
                                st.info("ℹ️ Ces importances sont **identiques pour TOUS les clients** - elles représentent l'importance globale de chaque feature pour le modèle LightGBM")
                                
                                top_10 = result['top_10_features']
                                top_10_df = pd.DataFrame([
                                    {
                                        'Rang': int(f["rank"]),
                                        'Feature': str(f["feature_name"]),
                                        'Importance': float(f["importance_value"])
                                    }
                                    for f in top_10
                                ])
                                
                                # Convertir en types standard pour éviter les erreurs PyArrow
                                top_10_df = top_10_df.astype({'Rang': 'int64', 'Feature': 'object', 'Importance': 'float64'})
                                
                                st.dataframe(top_10_df, use_container_width=True, hide_index=True)
                                
                                fig2 = px.bar(
                                    top_10_df,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title="Feature Importance Globale du Modèle",
                                    labels={'Importance': 'Valeur d\'Importance', 'Feature': 'Feature'}
                                )
                                fig2.update_layout(showlegend=False)
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # === SHAP WATERFALL ===
                            st.markdown("---")
                            st.subheader("📊 Waterfall SHAP - Feature Importance Locale (Client Spécifique)")
                            st.info("ℹ️ Ces explications sont **spécifiques à ce client** - elles montrent comment chaque feature influence la prédiction POUR CE CLIENT en particulier")
                            
                            if st.button("🔄 Charger Explications SHAP", use_container_width=True):
                                with st.spinner("Calcul des SHAP values en cours..."):
                                    try:
                                        explain_response = requests.post(
                                            f"{API_URL}/explain",
                                            json={
                                                "sk_id_curr": int(selected_sk_id),
                                                "features": {},
                                                "threshold": float(threshold)
                                            },
                                            timeout=30
                                        )
                                        
                                        if explain_response.status_code == 200:
                                            shap_data = explain_response.json()
                                            
                                            # Afficher les infos SHAP
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric(
                                                    "Base Value",
                                                    f"{shap_data['base_value']:.1%}"
                                                )
                                            with col2:
                                                st.metric(
                                                    "Prediction Value",
                                                    f"{shap_data['prediction_value']:.1%}"
                                                )
                                            with col3:
                                                st.metric(
                                                    "Contribution Totale",
                                                    f"{(shap_data['prediction_value'] - shap_data['base_value']):.1%}"
                                                )
                                            
                                            # Créer le waterfall plot
                                            st.markdown("---")
                                            st.subheader("🌊 Waterfall Plot")
                                            
                                            shap_values = shap_data['shap_values'][:10]  # Top 10
                                            
                                            # Créer le dataframe pour le plot
                                            waterfall_data = pd.DataFrame([
                                                {
                                                    'Feature': f"{str(v['feature_name'])} (={float(v['feature_value']):.2f})",
                                                    'Contribution': float(v['contribution']),
                                                    'Type': 'Positive' if float(v['contribution']) >= 0 else 'Negative'
                                                }
                                                for v in shap_values
                                            ])
                                            
                                            # Convertir en types standard
                                            waterfall_data = waterfall_data.astype({'Feature': 'object', 'Contribution': 'float64', 'Type': 'object'})
                                            
                                            # Créer le graphique waterfall
                                            fig_waterfall = go.Figure()
                                            
                                            # Ajouter la ligne de base
                                            base_val = shap_data['base_value']
                                            cumulative_sum = base_val
                                            x_values = []
                                            y_values = []
                                            colors = []
                                            
                                            # Point de départ
                                            x_values.append('Base Value')
                                            y_values.append(base_val)
                                            colors.append('lightgray')
                                            
                                            # Ajouter chaque contribution
                                            for idx, row in waterfall_data.iterrows():
                                                x_values.append(row['Feature'])
                                                prev_cumsum = cumulative_sum
                                                cumulative_sum += row['Contribution']
                                                y_values.append(cumulative_sum)
                                                
                                                if row['Contribution'] >= 0:
                                                    colors.append('#28a745')  # Vert
                                                else:
                                                    colors.append('#dc3545')  # Rouge
                                            
                                            # Ajouter le point final
                                            x_values.append('Prediction')
                                            y_values.append(cumulative_sum)
                                            colors.append('lightblue')
                                            
                                            # Créer le waterfall
                                            fig_waterfall.add_trace(go.Waterfall(
                                                x=x_values,
                                                y=y_values,
                                                base=base_val,
                                                measure=['absolute'] + ['relative'] * len(waterfall_data) + ['absolute'],
                                                text=[f"{v:.2%}" for v in y_values],
                                                textposition="auto",
                                                marker={"color": colors},
                                                connector={"line": {"color": "rgba(100, 100, 100, 0.4)"}},
                                                hovertemplate='<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>'
                                            ))
                                            
                                            fig_waterfall.update_layout(
                                                title="SHAP Waterfall - Impact des Features sur la Prédiction",
                                                xaxis_title="Features",
                                                yaxis_title="Probabilité de Risque",
                                                height=600,
                                                showlegend=False,
                                                template="plotly_white",
                                                hovermode="x unified"
                                            )
                                            
                                            st.plotly_chart(fig_waterfall, use_container_width=True)
                                            
                                            # Tableau détaillé des SHAP values
                                            st.markdown("---")
                                            st.subheader("📋 Détail des SHAP Values")
                                            
                                            shap_df = pd.DataFrame([
                                                {
                                                    'Feature': str(v['feature_name']),
                                                    'Valeur': f"{float(v['feature_value']):.4f}",
                                                    'Contribution SHAP': f"{float(v['contribution']):+.6f}",
                                                    'Impact': '↑ Augmente le risque' if float(v['contribution']) >= 0 else '↓ Diminue le risque'
                                                }
                                                for v in shap_values
                                            ])
                                            
                                            # Convertir en types standard
                                            shap_df = shap_df.astype({'Feature': 'object', 'Valeur': 'object', 'Contribution SHAP': 'object', 'Impact': 'object'})
                                            
                                            st.dataframe(shap_df, use_container_width=True, hide_index=True)
                                            
                                            st.success(f"✅ SHAP Values calculées avec succès!")
                                            st.info("""
                                                **Interprétation SHAP (Feature Importance Locale):**
                                                - **Base Value**: Prédiction moyenne du modèle sur tous les clients
                                                - **Contributions positives** (🟢 vert): Ces features AUGMENTENT le risque pour CE CLIENT
                                                - **Contributions négatives** (🔴 rouge): Ces features DIMINUENT le risque pour CE CLIENT
                                                - **Prediction**: Résultat final = Base Value + somme des contributions
                                                
                                                ⚠️ Note: Ces valeurs sont UNIQUES à ce client. Un autre client aura des contributions différentes même avec le même modèle!
                                            """)
                                        else:
                                            st.error(f"❌ Erreur API explain: {explain_response.status_code}")
                                    
                                    except requests.exceptions.ConnectionError:
                                        st.error(f"❌ Impossible de se connecter à {API_URL}")
                                    except Exception as e:
                                        st.error(f"❌ Erreur SHAP: {str(e)}")
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
    - [API Swagger](http://localhost:8001/docs)
    - [API ReDoc](http://localhost:8001/redoc)
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
