"""
Tracker MLflow pour le suivi des expériences de modèles
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import os
import tempfile


class MLFlowTracker:
    """Classe pour tracker les runs MLflow de manière simplifiée"""
    
    def __init__(self, experiment_name="Default", tracking_uri=None):
        """
        Initialise le tracker MLflow
        
        Args:
            experiment_name: Nom de l'expérience MLflow
            tracking_uri: URI du tracking (utilise un dossier temporaire si non spécifié)
        """
        self.experiment_name = experiment_name
        
        # Utiliser le dossier projet7-scoring-credit/mlruns pour le tracking
        # Remonter 3 niveaux depuis src/mlflow_tracking/tracker.py
        project_root = Path(__file__).parent.parent.parent
        tracking_path = project_root / "mlruns"
        
        print(f"📁 Tracking directory: {tracking_path}")
        
        # Créer le dossier s'il n'existe pas
        tracking_path.mkdir(parents=True, exist_ok=True)
        
        # Configurer MLflow avec le chemin au format URI file:///
        tracking_uri = tracking_path.as_uri()  # Convertit en file:///c:/...
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        
        # Désactiver les features problématiques de model registry
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
        
        # Créer ou set l'expérience
        try:
            mlflow.set_experiment(experiment_name)
            print(f"✅ Expérience configurée: {experiment_name}")
        except Exception as e:
            print(f"⚠️ Erreur: {e}")
        
        self.current_run = None
    
    def start_run(self, run_name=None):
        """Démarre un nouveau run MLflow"""
        try:
            self.current_run = mlflow.start_run(run_name=run_name)
        except:
            # Si un run est déjà actif, le fermer d'abord
            mlflow.end_run()
            self.current_run = mlflow.start_run(run_name=run_name)
        
        return self.current_run
    
    def end_run(self):
        """Termine le run courant"""
        if mlflow.active_run():
            mlflow.end_run()
        self.current_run = None
    
    def log_params(self, params_dict):
        """Log les paramètres"""
        for key, value in params_dict.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"Erreur lors du logging du paramètre {key}: {e}")
    
    def log_metrics(self, metrics_dict):
        """Log les métriques"""
        for key, value in metrics_dict.items():
            try:
                mlflow.log_metric(key, float(value))
            except Exception as e:
                print(f"Erreur lors du logging de la métrique {key}: {e}")
    
    def log_model(self, model, model_name="model"):
        """Log le modèle"""
        try:
            mlflow.sklearn.log_model(model, model_name)
        except Exception as e:
            print(f"Erreur lors du logging du modèle: {e}")
