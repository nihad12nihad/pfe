import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import json

class AgglomerativeClusteringModel:
    def __init__(self, n_clusters=3, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}

    def train(self, data_path):
        try:
            df = pd.read_csv(data_path)
            if df.empty or len(df) < self.n_clusters:
                raise ValueError("Données insuffisantes ou vides.")

            X = self.scaler.fit_transform(df.values)

            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters, linkage=self.linkage
            )
            labels = self.model.fit_predict(X)
            score = silhouette_score(X, labels)

            self.results = {
                "model": "AgglomerativeClustering",
                "parameters": {
                    "n_clusters": self.n_clusters,
                    "linkage": self.linkage
                },
                "silhouette_score": score,
                "labels": labels.tolist()
            }
            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="results/agglo_results.json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        return output_path

def run(data_path, n_clusters=3, linkage="ward"):
    model = AgglomerativeClusteringModel(n_clusters=n_clusters, linkage=linkage)
    results = model.train(data_path)
    return results
