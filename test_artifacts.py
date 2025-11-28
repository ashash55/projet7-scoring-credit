#!/usr/bin/env python3
"""
Test script pour déboguer la sauvegarde des artifacts MLflow
"""

import sys
import os
sys.path.insert(0, '.')

from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Charger le tracker
from src.mlflow_tracking.tracker import MLFlowTracker

# Créer un petit ensemble d'entraînement
X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

# Créer un pipeline simple
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
])

# Entraîner
pipe.fit(X, y)

# Initialiser le tracker
tracker = MLFlowTracker(experiment_name="test-artifacts", tracking_uri="./mlruns")

# Démarrer un run
tracker.start_run(run_name="test_artifact_logging")

# Log parameters
tracker.log_params({'test': 'value'})

# Log metrics
tracker.log_metrics({'test_metric': 0.95})

# Log model
print("\n📝 Tentative de logguer le modèle...")
tracker.log_model(pipe, model_name="test_model")

# Vérifier les artifacts
run_id = tracker.current_run.info.run_id
experiment_id = tracker.current_run.info.experiment_id

print(f"\n✅ Run ID: {run_id}")
print(f"✅ Experiment ID: {experiment_id}")

artifact_path = Path("./mlruns") / str(experiment_id) / run_id / "artifacts"
print(f"✅ Artifact path: {artifact_path}")

if artifact_path.exists():
    print(f"\n📂 Contenu du dossier artifacts:")
    for item in artifact_path.rglob("*"):
        rel_path = item.relative_to(artifact_path)
        size = item.stat().st_size if item.is_file() else "<dir>"
        print(f"  - {rel_path} ({size} bytes)" if isinstance(size, int) else f"  - {rel_path} (directory)")
else:
    print(f"\n❌ Le dossier artifacts n'existe pas!")

# Terminer le run
tracker.end_run()

print("\n✅ Test terminé!")
