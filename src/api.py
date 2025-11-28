"""
API de Prédiction de Crédit - LightGBM class_weight
Model: LightGBM avec stratégie class_weight
Seuil optimal: 0.46
Port: 8001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json
import mlflow
import mlflow.sklearn
import pickle
import os
import re
import uvicorn

# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALISATION FASTAPI
# ============================================================================
app = FastAPI(
    title="API de Prédiction de Crédit",
    description="API pour la prédiction de crédit utilisant LightGBM",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION MODÈLE
# ============================================================================
MLFLOW_TRACKING_URI = "notebooks/mlruns"
EXPERIMENT_NAME = "Default"  # Utilise l'expérience Default (experiment_id: 0)
OPTIMAL_THRESHOLD = 0.46
DEFAULT_THRESHOLD = 0.50

# Chemin vers les données light
DATA_LIGHT_PATH = "data/data_light_features.csv"

# Features du dataframe light (20 TOP features)
FEATURES_REQUIRED = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
    'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT'
]

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class ClientRequest(BaseModel):
    sk_id_curr: int
    features: Dict[str, float]
    threshold: Optional[float] = OPTIMAL_THRESHOLD

class FeatureImportance(BaseModel):
    feature_name: str
    importance_value: float
    rank: int

class PredictionResponse(BaseModel):
    sk_id_curr: int
    risk_probability: float
    decision: str  # "CRÉDIT ACCORDÉ" ou "CRÉDIT REFUSÉ"
    threshold_used: float
    distance_to_threshold: float
    top_10_features: List[FeatureImportance]
    model_info: Dict
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    threshold: float
    timestamp: str

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================
model = None
feature_importances = None
model_loaded = False
model_name = "LightGBM_class_weight"
df_light = None  # Cache pour le dataframe light

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes pour LightGBM"""
    forbidden_chars = r'[\[\]{}:,"\']'
    column_mapping = {}
    
    for col in df.columns:
        new_col = re.sub(forbidden_chars, '_', str(col))
        new_col = re.sub(r'_+', '_', new_col)
        new_col = new_col.strip('_')
        column_mapping[col] = new_col
    
    return df.rename(columns=column_mapping)

def load_model():
    """Charge le meilleur modèle LightGBM depuis MLflow"""
    global model, feature_importances, model_loaded, model_name
    
    try:
        logger.info("=" * 80)
        logger.info("🔄 Chargement du modèle LightGBM (LightGBM_class_weight)...")
        logger.info("=" * 80)
        
        # Le meilleur run trouvé après analyse (réentraîné 2025-11-28)
        BEST_RUN_ID = "90b4e1707c5d43eaa5d945213f437de5"
        BEST_RUN_NAME = "LightGBM_class_weight"
        BEST_F2_SCORE = 0.4115
        
        # Configurer MLflow avec le chemin local
        mlflow_path = Path("mlruns")
        mlflow.set_tracking_uri(str(mlflow_path))
        
        client = mlflow.tracking.MlflowClient(str(mlflow_path))
        
        # Chercher le run dans toutes les expériences
        best_run = None
        for exp in client.search_experiments():
            try:
                runs = client.search_runs(experiment_ids=[exp.experiment_id])
                for run in runs:
                    if run.info.run_id == BEST_RUN_ID:
                        best_run = run
                        break
                if best_run:
                    break
            except:
                continue
        
        if not best_run:
            logger.warning(f"⚠️  Run {BEST_RUN_ID} non trouvé")
            logger.warning("   Utilisation du modèle simulé pour les tests")
            model_loaded = False
            return False
        
        logger.info(f"✅ Run trouvé: {best_run.info.run_id}")
        logger.info(f"   Nom: {BEST_RUN_NAME}")
        logger.info(f"   F2-Score: {BEST_F2_SCORE}")
        
        # Charger le modèle
        try:
            # Chercher l'artifact du modèle
            artifacts_list = client.list_artifacts(best_run.info.run_id)
            logger.info(f"   Artifacts disponibles:")
            artifact_names = []
            for artifact in artifacts_list:
                logger.info(f"     - {artifact.path}")
                artifact_names.append(artifact.path)
            
            if not artifact_names:
                logger.warning(f"❌ Aucun artifact trouvé")
                model_loaded = False
                return False
            
            # Charger le premier artifact (le modèle)
            model_artifact_name = artifact_names[0]
            
            try:
                model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/{model_artifact_name}")
                logger.info(f"✅ Modèle chargé avec succès via MLflow ({model_artifact_name})!")
                model_name = BEST_RUN_NAME
            except Exception as e:
                logger.warning(f"⚠️  Impossible de charger via MLflow: {e}")
                
                # Essayer de charger depuis le disque directement
                artifact_path = mlflow_path / best_run.info.run_id / "artifacts" / model_artifact_name
                
                if artifact_path.exists():
                    try:
                        import joblib
                        model_pkl_path = artifact_path / "model.pkl"
                        if model_pkl_path.exists():
                            model = joblib.load(model_pkl_path)
                            logger.info(f"✅ Modèle chargé depuis le disque!")
                            model_name = BEST_RUN_NAME
                        else:
                            logger.warning(f"❌ Fichier model.pkl non trouvé")
                            model_loaded = False
                            return False
                    except Exception as e2:
                        logger.warning(f"❌ Erreur lors du chargement: {e2}")
                        model_loaded = False
                        return False
                else:
                    logger.warning(f"❌ Chemin artifact non trouvé: {artifact_path}")
                    model_loaded = False
                    return False
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            import traceback
            logger.error(traceback.format_exc())
            model_loaded = False
            return False
        
        # Extraire les feature importances
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importances = dict(zip(FEATURES_REQUIRED, model.feature_importances_))
                logger.info(f"✅ Feature importances extraites")
            elif hasattr(model, 'named_steps'):
                # Pipeline
                classifier = model.named_steps.get('classifier')
                if classifier and hasattr(classifier, 'feature_importances_'):
                    feature_importances = dict(zip(FEATURES_REQUIRED, classifier.feature_importances_))
                    logger.info(f"✅ Feature importances extraites du classifier")
            else:
                logger.warning(f"⚠️  Modèle sans feature_importances_")
                feature_importances = None
        except Exception as e:
            logger.warning(f"⚠️  Erreur extraction importances: {e}")
            feature_importances = None
        
        logger.info(f"✅ Modèle prêt pour les prédictions!")
        model_loaded = True
        return True
    
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("   Utilisation du modèle simulé")
        model_loaded = False
        return False
        model_loaded = False
        return False

def simulate_prediction(features: Dict[str, float]) -> tuple:
    """
    Simule une prédiction si le modèle réel n'est pas disponible
    Retourne (probabilité_risque, importances_features)
    """
    # Simulation simple basée sur les features
    risk_score = 0.5
    
    if 'EXT_SOURCE_2' in features:
        risk_score -= features['EXT_SOURCE_2'] * 0.25
    if 'DEBT_RATIO' in features:
        risk_score += features['DEBT_RATIO'] * 0.20
    if 'INSTAL_DAYS_PAST_DUE_MEAN' in features:
        risk_score += min(features['INSTAL_DAYS_PAST_DUE_MEAN'] / 100, 0.15)
    
    risk_prob = max(0, min(1, risk_score))
    
    # Créer des feature importances simulées
    simulated_importances = {
        'EXT_SOURCE_2': 0.20,
        'DEBT_RATIO': 0.18,
        'PAYMENT_RATE': 0.15,
        'INSTAL_DAYS_PAST_DUE_MEAN': 0.12,
        'AGE': 0.10,
        'CREDIT_DURATION': 0.08,
        'AMT_CREDIT': 0.07,
        'YEARS_EMPLOYED': 0.05,
        'BURO_AMT_CREDIT_SUM_DEBT_MEAN': 0.04,
        'CODE_GENDER': 0.01
    }
    
    return float(risk_prob), simulated_importances

def get_top_10_features(features_dict: Dict) -> List[FeatureImportance]:
    """Retourne les 10 features les plus importantes"""
    if not features_dict:
        features_dict = {}
    
    sorted_features = sorted(
        features_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return [
        FeatureImportance(
            feature_name=name,
            importance_value=float(value),
            rank=i+1
        )
        for i, (name, value) in enumerate(sorted_features)
    ]

# ============================================================================
# ENDPOINTS API
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global df_light
    logger.info("🚀 Démarrage API - Chargement des données light...")
    try:
        df_light = pd.read_csv(DATA_LIGHT_PATH)
        logger.info(f"✅ Données light chargées: {df_light.shape[0]} clients, {df_light.shape[1]} colonnes")
        # Définir SK_ID_CURR comme index si ce n'est pas déjà fait
        if 'SK_ID_CURR' in df_light.columns and df_light.index.name != 'SK_ID_CURR':
            df_light.set_index('SK_ID_CURR', inplace=True)
            logger.info(f"✅ SK_ID_CURR défini comme index")
    except Exception as e:
        logger.error(f"❌ Erreur chargement données light: {e}")
        df_light = None
    
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérifier la santé de l'API"""
    logger.info("✅ Health check demandé")
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=model_name,
        threshold=OPTIMAL_THRESHOLD,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(client_request: ClientRequest):
    """
    Prédire pour un client
    
    Paramètres:
    - sk_id_curr: Numéro client
    - features: Dictionnaire optionnel des features (si vide, récupère depuis df_light)
    - threshold: Seuil de décision (optionnel, défaut 0.46)
    """
    try:
        sk_id = client_request.sk_id_curr
        threshold = client_request.threshold or OPTIMAL_THRESHOLD
        
        logger.info(f"📊 Requête de prédiction pour le client {sk_id}")
        
        # Valider le seuil
        if not 0 <= threshold <= 1:
            raise HTTPException(
                status_code=400,
                detail="Le seuil doit être entre 0 et 1"
            )
        
        # Récupérer les features du client depuis df_light
        if df_light is None:
            raise HTTPException(
                status_code=503,
                detail="Données light non chargées"
            )
        
        if sk_id not in df_light.index:
            raise HTTPException(
                status_code=404,
                detail=f"Client {sk_id} non trouvé dans les données"
            )
        
        # Récupérer les features du client
        client_data = df_light.loc[sk_id]
        features = client_data.to_dict()
        
        logger.info(f"✅ Client trouvé: {len(features)} features")
        
        # Préparer les données pour la prédiction
        X = pd.DataFrame([features])
        X = clean_column_names(X)
        X = X[sorted(X.columns)]
        
        # Prédiction
        risk_prob = None
        importances = None
        
        if model_loaded and model is not None:
            try:
                logger.info(f"   Tentative prédiction avec le modèle réel...")
                # Essayer predict_proba
                try:
                    risk_prob = float(model.predict_proba(X)[0, 1])
                    logger.info(f"   ✅ predict_proba réussi")
                except:
                    # Essayer predict directement
                    logger.info(f"   ⚠️  predict_proba failed, trying predict()...")
                    pred = model.predict(X)
                    risk_prob = float(pred[0]) if len(pred.shape) == 1 else float(pred[0, 1])
                    logger.info(f"   ✅ predict réussi")
                
                # Utiliser les feature importances si disponibles
                importances = feature_importances if feature_importances else {}
                
            except Exception as e:
                logger.warning(f"⚠️  Erreur modèle réel: {type(e).__name__}: {e}")
                logger.warning(f"   Utilisation de la simulation")
                risk_prob, importances = simulate_prediction(features)
        else:
            logger.info(f"   Modèle non chargé, utilisation simulation")
            risk_prob, importances = simulate_prediction(features)
        
        # S'assurer que risk_prob est un float valide
        if risk_prob is None:
            risk_prob, importances = simulate_prediction(features)
        
        risk_prob = float(risk_prob)
        
        # Décision
        decision = "CRÉDIT REFUSÉ" if risk_prob >= threshold else "CRÉDIT ACCORDÉ"
        distance = abs(risk_prob - threshold)
        
        # Top 10 features
        top_10 = get_top_10_features(importances)
        
        logger.info(f"✅ Prédiction: {decision} (Risque: {risk_prob:.4f}, Seuil: {threshold})")
        
        return PredictionResponse(
            sk_id_curr=sk_id,
            risk_probability=risk_prob,
            decision=decision,
            threshold_used=threshold,
            distance_to_threshold=distance,
            top_10_features=top_10,
            model_info={
                'model_name': 'LightGBM',
                'strategy': 'class_weight',
                'f2_score': 0.4202,
                'recall': 0.6143,
                'precision': 0.1856
            },
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(clients: List[ClientRequest]):
    """Prédictions en batch pour plusieurs clients"""
    try:
        logger.info(f"📊 Batch request pour {len(clients)} clients")
        
        predictions = []
        errors = []
        
        for client in clients:
            try:
                # Réutiliser la logique de predict
                features = client.features
                threshold = client.threshold or OPTIMAL_THRESHOLD
                
                # Valider features
                missing = [f for f in FEATURES_REQUIRED if f not in features]
                if missing:
                    errors.append({
                        'sk_id_curr': client.sk_id_curr,
                        'error': f"Features manquantes: {missing}"
                    })
                    continue
                
                # Préparer données
                X = pd.DataFrame([features])
                X = clean_column_names(X)
                
                # Prédiction
                if model_loaded and model is not None:
                    try:
                        risk_prob = float(model.predict_proba(X)[0, 1])
                    except:
                        risk_prob, _ = simulate_prediction(features)
                else:
                    risk_prob, _ = simulate_prediction(features)
                
                decision = "CRÉDIT REFUSÉ" if risk_prob >= threshold else "CRÉDIT ACCORDÉ"
                
                predictions.append({
                    'sk_id_curr': client.sk_id_curr,
                    'risk_probability': risk_prob,
                    'decision': decision,
                    'threshold_used': threshold
                })
            
            except Exception as e:
                errors.append({
                    'sk_id_curr': client.sk_id_curr,
                    'error': str(e)
                })
        
        logger.info(f"✅ Batch traité: {len(predictions)} succès, {len(errors)} erreurs")
        
        return {
            'count': len(predictions),
            'predictions': predictions,
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Erreur batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clients")
async def list_clients():
    """Liste tous les clients disponibles"""
    global df_light
    
    if df_light is None:
        raise HTTPException(
            status_code=503,
            detail="Données light non chargées"
        )
    
    clients_list = df_light.index.tolist()
    return {
        'total_clients': len(clients_list),
        'clients': clients_list,
        'first_10': clients_list[:10],
        'last_10': clients_list[-10:]
    }

@app.get("/info")
async def model_info():
    """Obtenir les informations du modèle"""
    return {
        'model_name': 'LightGBM',
        'model_version': '1.0.0',
        'strategy': 'class_weight',
        'features_count': len(FEATURES_REQUIRED),
        'features': FEATURES_REQUIRED,
        'optimal_threshold': OPTIMAL_THRESHOLD,
        'default_threshold': DEFAULT_THRESHOLD,
        'data_source': 'data_light_features.csv',
        'total_clients': 307505,
        'metrics': {
            'f2_score': 0.4202,
            'recall': 0.6143,
            'precision': 0.1856,
            'accuracy': 0.7495,
            'auc': 0.7584
        },
        'created_at': '2025-11-26',
        'timestamp': datetime.now().isoformat()
    }

@app.get("/docs-html", response_class=HTMLResponse)
async def api_docs():
    """Documentation HTML de l'API"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API de Prédiction de Crédit</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 15px 0; border-left: 4px solid #3498db; border-radius: 5px; }
            code { background: #f4f4f4; padding: 3px 8px; border-radius: 3px; font-family: monospace; }
            pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .success { color: #27ae60; font-weight: bold; }
            .warning { color: #e67e22; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 API de Prédiction de Crédit</h1>
            <p><strong>Modèle:</strong> LightGBM avec stratégie class_weight</p>
            <p><strong>Seuil Optimal:</strong> 0.46 | <strong>F2-Score:</strong> 0.4202 | <strong>Recall:</strong> 0.6143</p>
            
            <h2>📋 Endpoints Disponibles</h2>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Vérifier la santé de l'API</p>
                <pre>curl http://localhost:8080/health</pre>
            </div>
            
            <div class="endpoint">
                <h3>POST /predict</h3>
                <p>Prédire pour un client</p>
                <pre>{
  "sk_id_curr": 100001,
  "features": { ...30 features... },
  "threshold": 0.46
}</pre>
                <p><strong>Réponse:</strong> Probabilité de risque, décision (CRÉDIT ACCORDÉ/REFUSÉ), top 10 features</p>
            </div>
            
            <div class="endpoint">
                <h3>POST /predict_batch</h3>
                <p>Prédictions en batch</p>
                <pre>[
  {"sk_id_curr": 100001, "features": {...}},
  {"sk_id_curr": 100002, "features": {...}}
]</pre>
            </div>
            
            <div class="endpoint">
                <h3>GET /info</h3>
                <p>Information du modèle et features</p>
                <pre>curl http://localhost:8080/info</pre>
            </div>
            
            <h2>⚙️ Configuration</h2>
            <ul>
                <li><strong>Seuil Optimal:</strong> 0.46 (minimise le score métier)</li>
                <li><strong>Seuil Défaut:</strong> 0.50</li>
                <li><strong>Features Required:</strong> 30 features numériques</li>
            </ul>
            
            <h2>✅ Status</h2>
            <p class="success">API Running on http://localhost:8080</p>
            <p class="warning">Pour les tests, utilisez le notebook Streamlit</p>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🚀 Démarrage de l'API de Prédiction de Crédit")
    logger.info("=" * 80)
    logger.info(f"📍 URL: http://localhost:8080")
    logger.info(f"📚 Docs: http://localhost:8080/docs")
    logger.info(f"🏥 Health: http://localhost:8080/health")
    logger.info(f"⚙️  Seuil Optimal: {OPTIMAL_THRESHOLD}")
    logger.info("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
