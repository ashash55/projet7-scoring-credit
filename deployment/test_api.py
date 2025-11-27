import pytest
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8001"

class TestAPIHealth:
    """Tests pour l'endpoint /health"""
    
    def test_health_status(self):
        response = requests.get(f"{API_URL}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
    
    def test_health_model_loaded(self):
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

class TestAPIInfo:
    """Tests pour l'endpoint /info"""
    
    def test_info_endpoint(self):
        response = requests.get(f"{API_URL}/info", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert data["model_name"] == "LightGBM"
    
    def test_info_features(self):
        response = requests.get(f"{API_URL}/info")
        data = response.json()
        assert "features" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0
    
    def test_info_metrics(self):
        response = requests.get(f"{API_URL}/info")
        data = response.json()
        assert "metrics" in data
        assert "auc" in data["metrics"]
        assert 0 <= data["metrics"]["auc"] <= 1

class TestAPIClients:
    """Tests pour l'endpoint /clients"""
    
    def test_clients_list(self):
        response = requests.get(f"{API_URL}/clients", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "total_clients" in data
        assert data["total_clients"] > 0
        assert "clients" in data
    
    def test_clients_count_307k(self):
        response = requests.get(f"{API_URL}/clients")
        data = response.json()
        assert data["total_clients"] == 307505

class TestAPIPredict:
    """Tests pour l'endpoint /predict"""
    
    def test_predict_valid_client(self):
        response = requests.post(
            f"{API_URL}/predict",
            json={"sk_id_curr": 100002, "features": {}, "threshold": 0.46},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "risk_probability" in data
        assert "decision" in data
        assert 0 <= data["risk_probability"] <= 1
    
    def test_predict_decision_format(self):
        response = requests.post(
            f"{API_URL}/predict",
            json={"sk_id_curr": 100002, "features": {}, "threshold": 0.46}
        )
        data = response.json()
        decision = data["decision"]
        assert decision in ["CRÉDIT ACCORDÉ", "CRÉDIT REFUSÉ"]
    
    def test_predict_invalid_client(self):
        response = requests.post(
            f"{API_URL}/predict",
            json={"sk_id_curr": 999999999, "features": {}, "threshold": 0.46}
        )
        assert response.status_code == 404
    
    def test_predict_threshold_validation(self):
        response = requests.post(
            f"{API_URL}/predict",
            json={"sk_id_curr": 100001, "features": {}, "threshold": 1.5}
        )
        assert response.status_code == 400
    
    def test_predict_consistency(self):
        # Même client, même seuil -> même prédiction
        payload = {"sk_id_curr": 100002, "features": {}, "threshold": 0.46}
        r1 = requests.post(f"{API_URL}/predict", json=payload)
        r2 = requests.post(f"{API_URL}/predict", json=payload)
        
        assert r1.json()["risk_probability"] == r2.json()["risk_probability"]

class TestAPIPerformance:
    """Tests de performance"""
    
    def test_health_response_time(self):
        import time
        start = time.time()
        requests.get(f"{API_URL}/health")
        elapsed = time.time() - start
        assert elapsed < 5.0  # < 5 seconds
    
    def test_predict_response_time(self):
        import time
        payload = {"sk_id_curr": 100001, "features": {}, "threshold": 0.46}
        start = time.time()
        requests.post(f"{API_URL}/predict", json=payload)
        elapsed = time.time() - start
        assert elapsed < 5.0  # < 5 seconds

class TestAPILogic:
    """Tests de logique métier"""
    
    def test_risk_probability_above_threshold_refusé(self):
        # Risque > seuil = REFUSÉ
        for _ in range(3):
            response = requests.post(
                f"{API_URL}/predict",
                json={"sk_id_curr": 100002, "features": {}, "threshold": 0.3}
            )
            data = response.json()
            if data["risk_probability"] >= 0.3:
                assert data["decision"] == "CRÉDIT REFUSÉ"
                break
    
    def test_risk_probability_below_threshold_accordé(self):
        # Risque < seuil = ACCORDÉ
        response = requests.post(
            f"{API_URL}/predict",
            json={"sk_id_curr": 100002, "features": {}, "threshold": 0.9}
        )
        data = response.json()
        if data["risk_probability"] <= 0.9:
            assert data["decision"] == "CRÉDIT ACCORDÉ"

def test_api_connectivity():
    """Test de connectivité basique"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        assert response.status_code in [200, 503]
    except requests.exceptions.ConnectionError:
        pytest.fail(f"Cannot connect to API at {API_URL}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
