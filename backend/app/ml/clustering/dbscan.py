import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
from pathlib import Path

class DBSCANClustering:
    """Classe pour appliquer DBSCAN Clustering"""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}

    def load_data(self, data_path):
        try:
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError("Le fichier de données est vide")
            return df
        except Exception as e:
            raise ValueError(f"Erreur de chargement des données : {str(e)}")

    def train(self, data_path):
        try:
            df = self.load_data(data_path)

            X_scaled = self.scaler.fit_transform(df)

            self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = self.model.fit_predict(X_scaled)

            unique_labels = set(labels)
            silhouette = silhouette_score(X_scaled, labels) if len(unique_labels) > 1 else -1

            self.results = {
                "model": "DBSCAN",
                "parameters": {
                    "eps": self.eps,
                    "min_samples": self.min_samples
                },
                "metrics": {"silhouette_score": silhouette},
                "labels": labels.tolist(),
            }
            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="results/dbscan_results.json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        return output_path

def run(data_path, eps=0.5, min_samples=5):
    model = DBSCANClustering(eps=eps, min_samples=min_samples)
    results = model.train(data_path)
    return results
