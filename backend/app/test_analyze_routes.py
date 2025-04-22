from fastapi.testclient import TestClient
from main import app  # Remplace par le nom réel
import os

client = TestClient(app)

def test_knn_analyze():
    # Exemple avec un fichier CSV fictif
    data_path = os.path.abspath("data/iris.csv")  # Mets ton vrai chemin
    response = client.post("/analyze", json={
        "algorithm": "knn",
        "data_path": data_path,
        "target_column": "species",  # Mets le vrai nom
        "parameters": {
            "n_neighbors": 3
        }
    })

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert json_data["algorithm"] == "knn"
    assert "result" in json_data

def test_invalid_algorithm():
    response = client.post("/analyze", json={
        "algorithm": "unknown_algo",
        "data_path": "some/path.csv",
        "parameters": {}
    })
    assert response.status_code == 400
    assert "n'est pas pris en charge" in response.json()["detail"]
