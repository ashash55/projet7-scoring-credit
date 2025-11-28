#!/usr/bin/env python3
"""
Script pour extraire le meilleur modèle LightGBM des runs MLflow
et le sauvegarder en pickle pour l'API
"""

import json
from pathlib import Path
import joblib
import mlflow

# Le run ID du meilleur modèle LightGBM_class_weight
run_id = "a3b2e2d84d8a4871abf79ff26a15b9b0"
experiment_id = "607710234398262663"

print(f"🔍 Recherche du modèle...")
print(f"   Run ID: {run_id}")
print(f"   Experiment ID: {experiment_id}")

# Chercher le modèle dans mlruns
mlruns_base = Path("mlruns") / experiment_id / run_id / "artifacts"

if mlruns_base.exists():
    print(f"✅ Dossier trouvé: {mlruns_base}")
    
    # Lister les artifacts
    artifacts = list(mlruns_base.glob("*"))
    print(f"✅ Artifacts: {[a.name for a in artifacts]}")
    
    # Chercher le modèle pkl
    for artifact_dir in artifacts:
        if artifact_dir.is_dir():
            pkl_file = artifact_dir / "model.pkl"
            if pkl_file.exists():
                print(f"✅ Modèle trouvé: {pkl_file}")
                
                # Charger et afficher les infos du modèle
                try:
                    model = joblib.load(pkl_file)
                    print(f"✅ Modèle chargé avec succès!")
                    print(f"   Type: {type(model)}")
                    
                    # Sauvegarder dans models/ pour l'API
                    models_dir = Path("models")
                    models_dir.mkdir(exist_ok=True)
                    output_path = models_dir / "lightgbm_class_weight.pkl"
                    joblib.dump(model, output_path)
                    print(f"✅ Modèle sauvegardé: {output_path}")
                    
                    # Créer un fichier de métadonnées
                    metadata = {
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "model_type": "LightGBM_class_weight",
                        "strategy": "class_weight",
                        "optimal_threshold": 0.46,
                        "metrics": {
                            "f2_score": 0.4202,
                            "recall": 0.6143,
                            "precision": 0.1856,
                            "accuracy": 0.7495,
                            "auc": 0.7584
                        }
                    }
                    
                    with open(models_dir / "model_metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                    print(f"✅ Métadonnées sauvegardées: {models_dir / 'model_metadata.json'}")
                    
                    print(f"\n{'='*70}")
                    print(f"🎉 Modèle extrait et sauvegardé avec succès!")
                    print(f"{'='*70}")
                    
                except Exception as e:
                    print(f"❌ Erreur lors du chargement: {e}")
                    import traceback
                    traceback.print_exc()
                break
else:
    print(f"❌ Dossier artifacts non trouvé: {mlruns_base}")
