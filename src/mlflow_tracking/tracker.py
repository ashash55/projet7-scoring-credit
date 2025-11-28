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
            # Toujours fermer tout run actif avant de commencer un nouveau
            if mlflow.active_run():
                try:
                    mlflow.end_run()
                except:
                    pass  # Ignorer les erreurs si le run n'existe plus
            
            # Attendre un peu pour éviter les conflits
            import time
            time.sleep(0.1)
            
            self.current_run = mlflow.start_run(run_name=run_name)
        except Exception as e:
            print(f"⚠️ Erreur lors du démarrage du run: {e}")
            # Essayer une deuxième fois après fermeture forcée
            try:
                if mlflow.active_run():
                    mlflow.end_run()
                self.current_run = mlflow.start_run(run_name=run_name)
            except Exception as e2:
                print(f"❌ Erreur persistante: {e2}")
                raise
        
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
        """Log le modèle avec gestion complète (sklearn + imblearn + joblib)"""
        import joblib
        import tempfile
        import shutil
        
        try:
            # Vérifier si c'est une imblearn Pipeline avec un classifieur
            classifier_to_log = model
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                # C'est une pipeline, extraire le classifieur
                classifier_to_log = model.named_steps['classifier']
                print(f"ℹ️ Pipeline détectée, extraction du classifieur: {type(classifier_to_log).__name__}")
            
            # Créer un dossier temporaire pour stocker le modèle
            temp_model_dir = tempfile.mkdtemp()
            model_pkl_path = os.path.join(temp_model_dir, "model.pkl")
            
            # Sauvegarder le modèle avec joblib
            joblib.dump(classifier_to_log, model_pkl_path)
            print(f"✅ Modèle sauvegardé en pickle: {model_pkl_path}")
            
            # Logguer le fichier pkl comme artifact directement dans le dossier "model"
            mlflow.log_artifact(model_pkl_path, artifact_path="model")
            print(f"✅ Modèle loggé en artifact MLflow")
            
            # Nettoyer les fichiers temporaires
            shutil.rmtree(temp_model_dir)
            
            # IMPORTANT: Terminer la run pour finalize les artifacts dans le filesystem
            # (sinon les artifacts restent vides)
            current_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "unknown"
            mlflow.end_run()
            print(f"✅ Run {current_run_id} terminée et artifacts finalisés")
            
        except Exception as e:
            print(f"❌ Erreur lors du logging du modèle: {e}")
            import traceback
            traceback.print_exc()
            if mlflow.active_run():
                mlflow.end_run()
