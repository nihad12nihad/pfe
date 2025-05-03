import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import json
from typing import Dict, Any

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

    def save_results(self, output_path="app/data/resultats/agglo_results.json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        return output_path

def run(X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    try:
        # Récupération des paramètres
        n_clusters = kwargs.get('n_clusters', 3)
        linkage = kwargs.get('linkage', 'ward')
        
        # Mise à l'échelle des données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values if hasattr(X, 'values') else X)
        
        # Modèle de clustering
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X_scaled)
        
        # Calcul du score de silhouette
        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1

        # Préparation des résultats
        results = {
            'metrics': {'silhouette_score': score},
            'cluster_labels': labels.tolist(),
            'model': 'AgglomerativeClustering',
            'parameters': {'n_clusters': n_clusters, 'linkage': linkage}
        }

        # Visualisation des données (si X_scaled a plus de 1 dimension)
        if X_scaled.shape[1] >= 2:
            results['visualization_data'] = X_scaled[:, :2].tolist()
        else:
            results['visualization_data'] = X_scaled.tolist()

        return results

    except Exception as e:
        raise RuntimeError(f"Erreur complète Agglomerative: {str(e)}")

