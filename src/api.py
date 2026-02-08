"""
API de Prédiction de Crédit - Interface Web Intégrée
Affichage des résultats directement dans FastAPI
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os
import re
import uvicorn
import shap
import json

# ============================================================================
# CONFIGURATION
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

app = FastAPI(
    title="API de Prédiction de Crédit",
    description="API avec interface web intégrée",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPTIMAL_THRESHOLD = 0.46
DATA_LIGHT_PATH = "data/data_mini_features.csv"

FEATURES_REQUIRED = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
    'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT'
]

# ============================================================================
# MODELS
# ============================================================================
class ClientRequest(BaseModel):
    sk_id_curr: int
    features: Dict[str, float] = {}
    threshold: Optional[float] = OPTIMAL_THRESHOLD

class FeatureImportance(BaseModel):
    feature_name: str
    importance_value: float
    rank: int

class ShapValue(BaseModel):
    feature_name: str
    contribution: float
    feature_value: float

class ExplanationResponse(BaseModel):
    sk_id_curr: int
    base_value: float
    prediction_value: float
    shap_values: List[ShapValue]
    feature_values: Dict[str, float]

class PredictionResponse(BaseModel):
    sk_id_curr: int
    risk_probability: float
    decision: str
    threshold_used: float
    distance_to_threshold: float
    top_10_features: List[FeatureImportance]
    model_info: Dict
    timestamp: str

# ============================================================================
# GLOBALS
# ============================================================================
model = None
feature_importances = None
model_loaded = False
df_light = None
explainer = None
X_background = None  # Pour SHAP TreeExplainer ou KernelExplainer

# ============================================================================
# UTILITIES
# ============================================================================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
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
    global model, feature_importances, model_loaded, model_name, explainer, X_background
    
    try:
        logger.info("🔄 Chargement du modèle LightGBM...")
        
        BEST_RUN_ID = "f896836ac5f24fc7afce8af71c9bdc3a"
        
        # Chercher mlruns
        mlflow_path = None
        for potential_path in [
            Path("./mlruns"),
            Path("mlruns"),
            Path("../mlruns"),
            Path(__file__).parent.parent / "mlruns",
        ]:
            if potential_path.exists():
                mlflow_path = potential_path
                logger.info(f"✅ MLflow trouvé: {mlflow_path}")
                break
        
        if mlflow_path is None:
            logger.warning("⚠️ MLflow non trouvé, mode simulation")
            model_loaded = False
            return False
        
        # Chercher le fichier model.pkl directement
        model_file = mlflow_path / "721715311403030274" / BEST_RUN_ID / "artifacts" / "model" / "model.pkl"
        
        logger.info(f"📂 Chemin modèle: {model_file}")
        
        if model_file.exists():
            logger.info("✅ Fichier model.pkl trouvé!")
            
            # Charger avec joblib
            import joblib
            model = joblib.load(model_file)
            logger.info("✅ Modèle chargé avec succès!")
            
            # Extraire feature importances
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_importances = dict(zip(FEATURES_REQUIRED, model.feature_importances_))
                    logger.info("✅ Feature importances extraites")
                elif hasattr(model, 'named_steps'):
                    classifier = model.named_steps.get('classifier')
                    if classifier and hasattr(classifier, 'feature_importances_'):
                        feature_importances = dict(zip(FEATURES_REQUIRED, classifier.feature_importances_))
                        logger.info("✅ Feature importances extraites du pipeline")
                else:
                    logger.warning("⚠️ Pas de feature importances")
                    feature_importances = None
            except Exception as e:
                logger.warning(f"⚠️ Erreur extraction importances: {e}")
                feature_importances = None
            
            model_loaded = True
            model_name = "LightGBM_class_weight"
            logger.info("✅ Modèle prêt pour les prédictions!")
            return True
        else:
            logger.error(f"❌ Fichier non trouvé: {model_file}")
            logger.warning("⚠️ Mode simulation activé")
            model_loaded = False
            return False
    
    except Exception as e:
        logger.error(f"❌ Erreur: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        model_loaded = False
        return False

def initialize_shap_explainer():
    """Initialise l'explainer SHAP avec un sous-ensemble des données"""
    global explainer, X_background, df_light
    
    try:
        if model is None or df_light is None:
            logger.warning("⚠️ Modèle ou données non chargées, SHAP non initialisé")
            return False
        
        logger.info("🔄 Initialisation SHAP Explainer...")
        
        # Utiliser un sous-ensemble aléatoire pour le background
        sample_size = min(100, len(df_light))
        X_background = df_light.sample(n=sample_size, random_state=42)
        X_background = clean_column_names(X_background)
        
        # Créer l'explainer SHAP TreeExplainer pour LightGBM
        try:
            explainer = shap.TreeExplainer(model)
            logger.info(f"✅ SHAP TreeExplainer initialisé avec {sample_size} exemples")
            return True
        except Exception as e:
            logger.warning(f"⚠️ TreeExplainer échoué: {e}, utilisation KernelExplainer")
            # Fallback sur KernelExplainer si TreeExplainer échoue
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1],
                X_background
            )
            logger.info("✅ SHAP KernelExplainer initialisé")
            return True
    
    except Exception as e:
        logger.error(f"❌ Erreur SHAP: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def simulate_prediction(features: Dict[str, float]) -> tuple:
    """
    Simule une prédiction avec importances dynamiques basées sur les features du client.
    Les importances varient selon les valeurs spécifiques du client.
    """
    # Prédiction basée sur quelques features clés
    risk_score = 0.5
    if 'EXT_SOURCE_2' in features:
        risk_score -= features.get('EXT_SOURCE_2', 0) * 0.25
    if 'DEBT_RATIO' in features:
        risk_score += features.get('DEBT_RATIO', 0) * 0.20
    
    risk_prob = max(0, min(1, risk_score))
    
    # Calculer les importances DYNAMIQUES basées sur les features du client
    # Stratégie: importance = valeur normalisée + écart à la moyenne
    
    # Définir les moyennes approximatives pour chaque feature
    feature_means = {
        'EXT_SOURCE_2': 0.5,
        'DEBT_RATIO': 0.5,
        'PAYMENT_RATE': 0.3,
        'INSTAL_DAYS_PAST_DUE_MEAN': 10.0,
        'AGE': 40.0,
        'AMT_CREDIT': 100000.0,
        'AMT_INCOME_TOTAL': 150000.0,
        'DAYS_EMPLOYED': 1000.0,
        'DAYS_BIRTH': 12000.0,
        'EXT_SOURCE_1': 0.5
    }
    
    # Calculer les écarts normalisés pour chaque feature
    importance_dict = {}
    
    for feature_name, feature_value in features.items():
        if feature_name in feature_means:
            mean_val = feature_means[feature_name]
            
            # Éviter division par zéro
            if mean_val != 0:
                # Calcul d'importance basée sur l'écart à la moyenne
                # Plus l'écart est grand, plus l'importance est élevée
                ratio = abs(feature_value - mean_val) / max(abs(mean_val), 1.0)
                
                # Normaliser entre 0 et 1
                importance = min(0.3, ratio * 0.1)  # Capper à 0.3
            else:
                importance = abs(feature_value) * 0.05
        else:
            # Pour les autres features, importance basée sur la valeur absolue normalisée
            importance = min(0.15, abs(feature_value) * 0.001)
        
        if importance > 0.01:  # Seuil minimum
            importance_dict[feature_name] = importance
    
    # Normaliser les importances pour qu'elles somment à near 1
    total_importance = sum(importance_dict.values())
    if total_importance > 0:
        importance_dict = {k: v / total_importance for k, v in importance_dict.items()}
    else:
        # Fallback: importances de secours
        importance_dict = {
            'EXT_SOURCE_2': 0.20,
            'DEBT_RATIO': 0.18,
            'PAYMENT_RATE': 0.15,
            'INSTAL_DAYS_PAST_DUE_MEAN': 0.12,
            'AGE': 0.10
        }
    
    return float(risk_prob), importance_dict

def get_top_10_features(features_dict: Dict) -> List[FeatureImportance]:
    if not features_dict:
        features_dict = {}
    
    sorted_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return [
        FeatureImportance(
            feature_name=name,
            importance_value=float(value),
            rank=i+1
        )
        for i, (name, value) in enumerate(sorted_features)
    ]

# ============================================================================
# STARTUP
# ============================================================================
@app.on_event("startup")
async def startup_event():
    global df_light
    logger.info("🚀 Démarrage API...")
    
    possible_paths = [
        "data/data_mini_features.csv",
        "./data/data_mini_features.csv",
        "/app/data/data_mini_features.csv",
        Path(__file__).parent.parent / "data" / "data_mini_features.csv",
    ]
    
    data_path = None
    for path in possible_paths:
        path_obj = Path(path) if isinstance(path, str) else path
        if path_obj.exists():
            data_path = str(path_obj)
            break
    
    if data_path:
        try:
            df_light = pd.read_csv(data_path, index_col='SK_ID_CURR')
            logger.info(f"✅ Données chargées: {df_light.shape}")
        except Exception as e:
            logger.error(f"❌ Erreur données: {e}")
            df_light = None
    
    load_model()
    initialize_shap_explainer()

# ============================================================================
# INTERFACE WEB PRINCIPALE
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def home():
    """Interface web principale"""
    html = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prédiction de Crédit - Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .status-bar {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .status-item {
            text-align: center;
            padding: 10px 20px;
        }
        
        .status-item label {
            display: block;
            color: #888;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .status-item value {
            display: block;
            color: #333;
            font-size: 1.3em;
            font-weight: bold;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background: #27ae60;
            box-shadow: 0 0 10px #27ae60;
        }
        
        .status-offline {
            background: #e74c3c;
        }
        
        .main-content {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .section {
            margin-bottom: 30px;
        }
        
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            color: #555;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-secondary {
            background: #95a5a6;
            margin-left: 10px;
        }
        
        #result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 10px;
            display: none;
        }
        
        .result-approved {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 2px solid #28a745;
        }
        
        .result-rejected {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border: 2px solid #dc3545;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .result-decision {
            font-size: 2em;
            font-weight: bold;
        }
        
        .decision-approved {
            color: #28a745;
        }
        
        .decision-rejected {
            color: #dc3545;
        }
        
        .result-probability {
            font-size: 1.5em;
        }
        
        .metric {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        
        .features-table {
            width: 100%;
            margin-top: 20px;
            border-collapse: collapse;
        }
        
        .features-table th, .features-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .features-table th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }
        
        .features-table tr:hover {
            background: #f8f9fa;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #dc3545);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .clients-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .client-badge {
            background: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
        }
        
        .client-badge:hover {
            background: #667eea;
            color: white;
            transform: scale(1.05);
            border-color: #667eea;
        }
        
        .waterfall-container {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 2px solid #667eea;
        }
        
        .waterfall-title {
            color: #667eea;
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        .waterfall-plot {
            width: 100%;
            height: 500px;
            border-radius: 8px;
            background: white;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .waterfall-bar {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            padding: 8px;
            background: white;
            border-radius: 5px;
        }
        
        .waterfall-label {
            min-width: 150px;
            font-weight: 600;
            font-size: 0.9em;
            color: #333;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .waterfall-bar-container {
            flex: 1;
            height: 25px;
            background: #e9ecef;
            border-radius: 5px;
            margin: 0 10px;
            position: relative;
            overflow: hidden;
        }
        
        .waterfall-bar-positive {
            background: linear-gradient(90deg, #28a745, #20c997);
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s ease;
        }
        
        .waterfall-bar-negative {
            background: linear-gradient(90deg, #dc3545, #fd7e14);
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s ease;
        }
        
        .waterfall-value {
            min-width: 80px;
            text-align: right;
            font-weight: 600;
            font-size: 0.9em;
            color: #333;
        }
        
        .waterfall-legend {
            display: flex;
            gap: 30px;
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .legend-box {
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .status-bar {
                flex-direction: column;
            }
            
            .result-header {
                flex-direction: column;
                gap: 15px;
            }
            
            .waterfall-label {
                min-width: 100px;
                font-size: 0.8em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🎯 Prédiction de Crédit</h1>
            <p>Système d'analyse de risque de crédit avec LightGBM</p>
        </div>
        
        <!-- Status Bar -->
        <div class="status-bar" id="statusBar">
            <div class="status-item">
                <label>Status API</label>
                <value>
                    <span class="status-indicator status-offline"></span>
                    <span id="apiStatus">Chargement...</span>
                </value>
            </div>
            <div class="status-item">
                <label>Modèle</label>
                <value id="modelName">-</value>
            </div>
            <div class="status-item">
                <label>Seuil Optimal</label>
                <value id="threshold">-</value>
            </div>
            <div class="status-item">
                <label>Clients Disponibles</label>
                <value id="clientsCount">-</value>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <div class="section">
                <h2>🔍 Sélection du Client</h2>
                
                <div class="form-group">
                    <label for="clientId">ID Client</label>
                    <input type="number" id="clientId" placeholder="Ex: 100001">
                </div>
                
                <div class="form-group">
                    <label for="thresholdInput">Seuil de Décision (0-1)</label>
                    <input type="number" id="thresholdInput" step="0.01" min="0" max="1" value="0.46">
                </div>
                
                <div>
                    <button class="btn" onclick="predict()">🚀 Analyser</button>
                    <button class="btn btn-secondary" onclick="loadClients()">📋 Voir Clients</button>
                    <button class="btn btn-secondary" onclick="clearResult()">🗑️ Effacer</button>
                </div>
            </div>
            
            <!-- Clients List -->
            <div class="section" id="clientsSection" style="display:none;">
                <h2>📋 Liste des Clients</h2>
                <div class="clients-grid" id="clientsList"></div>
            </div>
            
            <!-- Loading -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyse en cours...</p>
            </div>
            
            <!-- Result -->
            <div id="result">
                <div class="result-header">
                    <div>
                        <div class="result-decision" id="decision">-</div>
                        <div style="color: #666; margin-top: 5px;">Client #<span id="resultClientId">-</span></div>
                    </div>
                    <div class="result-probability" id="probability">-</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Probabilité de Risque</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressBar">0%</div>
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Distance au Seuil</div>
                    <div class="metric-value" id="distance">-</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Seuil Utilisé</div>
                    <div class="metric-value" id="usedThreshold">-</div>
                </div>
                
                <h3 style="margin-top: 30px; color: #667eea;">🔝 Top 10 Features Importantes</h3>
                <table class="features-table">
                    <thead>
                        <tr>
                            <th>Rang</th>
                            <th>Feature</th>
                            <th>Importance</th>
                        </tr>
                    </thead>
                    <tbody id="featuresTable"></tbody>
                </table>
                
                <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    <strong>Informations Modèle:</strong>
                    <div id="modelInfo" style="margin-top: 10px; color: #666;"></div>
                </div>
                
                <!-- Waterfall SHAP Section -->
                <div class="waterfall-container" id="waterfallSection" style="display: none;">
                    <div class="waterfall-title">📊 Waterfall SHAP - Explication Locale</div>
                    <button class="btn" onclick="loadWaterfall()">🔄 Charger Waterfall</button>
                    <div class="loading" id="waterfallLoading" style="margin-top: 20px; display: none;">
                        <div class="spinner"></div>
                        <p>Calcul des SHAP values...</p>
                    </div>
                    <div class="waterfall-plot" id="waterfallPlot" style="display: none;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Check API health on load
        window.onload = function() {
            checkHealth();
        };
        
        async function checkHealth() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                
                document.getElementById('apiStatus').textContent = data.status === 'healthy' ? 'En ligne' : 'Dégradé';
                document.querySelector('.status-indicator').className = 
                    'status-indicator ' + (data.status === 'healthy' ? 'status-online' : 'status-offline');
                document.getElementById('modelName').textContent = data.model_name;
                document.getElementById('threshold').textContent = data.threshold;
                
                // Get clients count
                const clientsResponse = await fetch('/clients');
                const clientsData = await clientsResponse.json();
                document.getElementById('clientsCount').textContent = clientsData.total_clients;
                
            } catch (error) {
                console.error('Error checking health:', error);
                document.getElementById('apiStatus').textContent = 'Erreur';
            }
        }
        
        async function predict() {
            const clientId = parseInt(document.getElementById('clientId').value);
            const threshold = parseFloat(document.getElementById('thresholdInput').value);
            
            if (!clientId) {
                alert('Veuillez entrer un ID client');
                return;
            }
            
            if (threshold < 0 || threshold > 1) {
                alert('Le seuil doit être entre 0 et 1');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        sk_id_curr: clientId,
                        features: {},
                        threshold: threshold
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Erreur de prédiction');
                }
                
                const data = await response.json();
                displayResult(data);
                
            } catch (error) {
                alert('Erreur: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayResult(data) {
            // Show result section
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            
            // Afficher la section waterfall
            document.getElementById('waterfallSection').style.display = 'block';
            
            // Set decision styling
            const isApproved = data.decision === 'CRÉDIT ACCORDÉ';
            resultDiv.className = isApproved ? 'result-approved' : 'result-rejected';
            
            // Update decision
            const decisionEl = document.getElementById('decision');
            decisionEl.textContent = data.decision;
            decisionEl.className = 'result-decision ' + (isApproved ? 'decision-approved' : 'decision-rejected');
            
            // Update values
            document.getElementById('resultClientId').textContent = data.sk_id_curr;
            document.getElementById('probability').textContent = 
                'Risque: ' + (data.risk_probability * 100).toFixed(2) + '%';
            document.getElementById('distance').textContent = 
                (data.distance_to_threshold * 100).toFixed(2) + '%';
            document.getElementById('usedThreshold').textContent = data.threshold_used;
            
            // Update progress bar
            const progressBar = document.getElementById('progressBar');
            const percentage = (data.risk_probability * 100).toFixed(1);
            progressBar.style.width = percentage + '%';
            progressBar.textContent = percentage + '%';
            
            // Update features table
            const tableBody = document.getElementById('featuresTable');
            tableBody.innerHTML = '';
            data.top_10_features.forEach(feature => {
                const row = tableBody.insertRow();
                row.innerHTML = `
                    <td>${feature.rank}</td>
                    <td>${feature.feature_name}</td>
                    <td>${feature.importance_value.toFixed(4)}</td>
                `;
            });
            
            // Update model info
            const modelInfo = document.getElementById('modelInfo');
            modelInfo.innerHTML = `
                <div>Modèle: ${data.model_info.model_name}</div>
                <div>Stratégie: ${data.model_info.strategy}</div>
                <div>F2-Score: ${data.model_info.f2_score}</div>
                <div>Recall: ${data.model_info.recall}</div>
                <div>Precision: ${data.model_info.precision}</div>
            `;
            
            // Scroll to result
            resultDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
        async function loadClients() {
            try {
                const response = await fetch('/clients');
                const data = await response.json();
                
                const clientsList = document.getElementById('clientsList');
                const clientsSection = document.getElementById('clientsSection');
                
                clientsList.innerHTML = '';
                data.first_10.forEach(clientId => {
                    const badge = document.createElement('div');
                    badge.className = 'client-badge';
                    badge.textContent = clientId;
                    badge.onclick = () => {
                        document.getElementById('clientId').value = clientId;
                        clientsSection.style.display = 'none';
                    };
                    clientsList.appendChild(badge);
                });
                
                clientsSection.style.display = 'block';
                clientsSection.scrollIntoView({ behavior: 'smooth' });
                
            } catch (error) {
                alert('Erreur chargement clients: ' + error.message);
            }
        }
        
        function clearResult() {
            document.getElementById('result').style.display = 'none';
            document.getElementById('clientId').value = '';
            document.getElementById('clientsSection').style.display = 'none';
            document.getElementById('waterfallSection').style.display = 'none';
        }
        
        async function loadWaterfall() {
            const clientId = parseInt(document.getElementById('clientId').value);
            
            if (!clientId) {
                alert('Veuillez d\'abord analyser un client');
                return;
            }
            
            document.getElementById('waterfallLoading').style.display = 'block';
            document.getElementById('waterfallPlot').style.display = 'none';
            
            try {
                const response = await fetch('/explain', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        sk_id_curr: clientId,
                        features: {},
                        threshold: 0.46
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Erreur calcul SHAP');
                }
                
                const data = await response.json();
                displayWaterfall(data);
                
            } catch (error) {
                alert('Erreur: ' + error.message);
            } finally {
                document.getElementById('waterfallLoading').style.display = 'none';
            }
        }
        
        function displayWaterfall(data) {
            const waterfallPlot = document.getElementById('waterfallPlot');
            waterfallPlot.innerHTML = '';
            
            // Trier par contribution absolue et ne garder que le top 10
            const topShapValues = data.shap_values.slice(0, 10);
            
            // Créer le graphique waterfall
            const waterfallHtml = topShapValues.map(shap => {
                const isPositive = shap.contribution >= 0;
                const width = Math.abs(shap.contribution) * 100; // Normaliser pour affichage
                const barClass = isPositive ? 'waterfall-bar-positive' : 'waterfall-bar-negative';
                const label = `${shap.feature_name} (${shap.feature_value.toFixed(2)})`;
                
                return `
                    <div class="waterfall-bar">
                        <div class="waterfall-label" title="${label}">${label}</div>
                        <div class="waterfall-bar-container">
                            <div class="${barClass}" style="width: ${Math.min(width, 100)}%;"></div>
                        </div>
                        <div class="waterfall-value">${shap.contribution > 0 ? '+' : ''}${shap.contribution.toFixed(4)}</div>
                    </div>
                `;
            }).join('');
            
            const legendHtml = `
                <div class="waterfall-legend">
                    <div style="padding: 10px; background: white; border-radius: 5px;">
                        <strong>Prédiction:</strong> ${(data.prediction_value * 100).toFixed(2)}%
                    </div>
                    <div style="padding: 10px; background: white; border-radius: 5px;">
                        <strong>Base Value:</strong> ${(data.base_value * 100).toFixed(2)}%
                    </div>
                </div>
            `;
            
            waterfallPlot.innerHTML = waterfallHtml + legendHtml;
            waterfallPlot.style.display = 'block';
        }
    </script>
</body>
</html>
    """
    return html

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_name": "LightGBM_class_weight",
        "threshold": OPTIMAL_THRESHOLD,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(client_request: ClientRequest):
    try:
        sk_id = client_request.sk_id_curr
        threshold = client_request.threshold or OPTIMAL_THRESHOLD
        
        if not 0 <= threshold <= 1:
            raise HTTPException(status_code=400, detail="Seuil entre 0 et 1")
        
        if df_light is None:
            raise HTTPException(status_code=503, detail="Données non chargées")
        
        if sk_id not in df_light.index:
            raise HTTPException(status_code=404, detail=f"Client {sk_id} non trouvé")
        
        client_data = df_light.loc[sk_id]
        features = client_data.to_dict()
        
        X = pd.DataFrame([features])
        X = clean_column_names(X)
        X = X[sorted(X.columns)]
        
        risk_prob = None
        importances = None
        
        if model_loaded and model is not None:
            try:
                # Prédiction
                risk_prob = float(model.predict_proba(X)[0, 1])

                # Essayer d'extraire les importances spécifiques au client via SHAP
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)

                    # shap_values peut être une liste (par classe) ou un array
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        shap_client = shap_values[1][0]
                    else:
                        # fallback: prendre la première dimension
                        shap_client = np.array(shap_values)[0]
                        if shap_client.ndim > 1:
                            shap_client = shap_client[0]

                    top_features_idx = np.argsort(-np.abs(shap_client))[:10]
                    importances = {X.columns[i]: float(shap_client[i]) for i in top_features_idx}
                except Exception:
                    # Si SHAP échoue, fallback sur importances globales si présentes
                    importances = feature_importances if feature_importances else {}

            except Exception:
                risk_prob, importances = simulate_prediction(features)
        else:
            risk_prob, importances = simulate_prediction(features)
        
        if risk_prob is None:
            risk_prob, importances = simulate_prediction(features)
        
        risk_prob = float(risk_prob)
        decision = "CRÉDIT REFUSÉ" if risk_prob >= threshold else "CRÉDIT ACCORDÉ"
        distance = abs(risk_prob - threshold)
        top_10 = get_top_10_features(importances)
        
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
        logger.error(f"❌ Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain(client_request: ClientRequest):
    """Retourne les SHAP values pour l'explication locale du client"""
    try:
        sk_id = client_request.sk_id_curr
        
        if df_light is None:
            raise HTTPException(status_code=503, detail="Données non chargées")
        
        if sk_id not in df_light.index:
            raise HTTPException(status_code=404, detail=f"Client {sk_id} non trouvé")
        
        if not model_loaded or model is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        
        if explainer is None:
            raise HTTPException(status_code=503, detail="Explainer SHAP non initialisé")
        
        # Récupérer les données du client
        client_data = df_light.loc[sk_id]
        features = client_data.to_dict()
        
        X = pd.DataFrame([features])
        X = clean_column_names(X)
        X = X[sorted(X.columns)]
        
        # Calculer les SHAP values
        logger.info(f"⏳ Calcul SHAP pour client {sk_id}...")
        shap_values = explainer.shap_values(X)
        
        # Gérer les cas où shap_values est une liste (pour la classe 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Obtenir la prédiction
        risk_prob = float(model.predict_proba(X)[0, 1])
        
        # Extraire les base values (expected model output)
        if hasattr(explainer, 'expected_value'):
            if isinstance(explainer.expected_value, list):
                base_value = float(explainer.expected_value[1])
            else:
                base_value = float(explainer.expected_value)
        else:
            base_value = 0.5
        
        # Préparer les résultats
        feature_names = X.columns.tolist()
        feature_values = X.iloc[0].to_dict()
        
        shap_list = []
        for i, feature_name in enumerate(feature_names):
            shap_contribution = float(shap_values[0, i]) if isinstance(shap_values, np.ndarray) else float(shap_values[i])
            shap_list.append(ShapValue(
                feature_name=feature_name,
                contribution=shap_contribution,
                feature_value=float(feature_values.get(feature_name, 0))
            ))
        
        # Trier par contribution absolue
        shap_list.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        logger.info(f"✅ SHAP calculé pour client {sk_id}")
        
        return ExplanationResponse(
            sk_id_curr=sk_id,
            base_value=base_value,
            prediction_value=risk_prob,
            shap_values=shap_list,
            feature_values=feature_values
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur SHAP: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clients")
async def list_clients():
    if df_light is None:
        raise HTTPException(status_code=503, detail="Données non chargées")
    
    clients_list = df_light.index.tolist()
    return {
        'total_clients': len(clients_list),
        'clients': clients_list,
        'first_10': clients_list[:10],
        'last_10': clients_list[-10:]
    }

@app.get("/info")
async def model_info():
    return {
        'model_name': 'LightGBM',
        'model_version': '1.0.0',
        'strategy': 'class_weight',
        'features_count': len(FEATURES_REQUIRED),
        'optimal_threshold': OPTIMAL_THRESHOLD,
        'metrics': {
            'f2_score': 0.4202,
            'recall': 0.6143,
            'precision': 0.1856
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8001))
    logger.info("🚀 API avec Interface Web")
    logger.info(f"📍 URL: http://localhost:{port}")
    logger.info(f"🌐 Interface: http://localhost:{port}/")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")