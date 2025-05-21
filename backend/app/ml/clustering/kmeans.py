import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
from pathlib import Path
from typing import Dict, Any


class KMeansClustering:
    """Classe pour appliquer KMeans Clustering avec évaluation et sauvegarde des résultats."""

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Charge le jeu de données depuis un fichier CSV."""
        try:
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError("Le fichier de données est vide")
            return df
        except Exception as e:
            raise ValueError(f"Erreur de chargement des données : {str(e)}")

    def train(self, data_path: str) -> Dict[str, Any]:
        """Entraîne le modèle KMeans sur les données fournies."""
        try:
            df = self.load_data(data_path)

            if df.isnull().any().any():
                raise ValueError("Les données contiennent des valeurs manquantes.")

            X_scaled = self.scaler.fit_transform(df)

            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.model.fit(X_scaled)

            labels = self.model.labels_
            silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1

            self.results = {
                "model": "KMeans",
                "parameters": {"n_clusters": self.n_clusters},
                "metrics": {"silhouette_score": silhouette},
                "cluster_centers": self.model.cluster_centers_.tolist(),
                "labels": labels.tolist(),
                "visualization_data": X_scaled[:, :2].tolist() if X_scaled.shape[1] >= 2 else X_scaled.tolist()
            }
            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="app/data/resultats/kmeans_results.json") -> str:
        """Sauvegarde les résultats dans un fichier JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        return output_path


def run(X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Fonction utilitaire pour exécuter KMeans en mode programmatique."""
    try:
        n_clusters = kwargs.get('n_clusters', 3)

        if X.isnull().any().any():
            raise ValueError("Les données contiennent des valeurs manquantes.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1

        return {
            'metrics': {'silhouette_score': score},
            'cluster_labels': labels.tolist(),
            'visualization_data': X_scaled[:, :2].tolist() if X_scaled.shape[1] >= 2 else X_scaled.tolist(),
            'model': 'KMeans',
            'parameters': {'n_clusters': n_clusters}
        }

    except Exception as e:
        raise RuntimeError(f"Erreur complète KMeans: {e}")
