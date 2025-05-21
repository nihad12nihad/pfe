import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import json
from typing import Dict, Any


class AgglomerativeClusteringModel:
    """Classe pour appliquer Agglomerative Clustering avec évaluation et sauvegarde."""

    def __init__(self, n_clusters=3, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Charge et valide les données depuis un fichier CSV."""
        try:
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError("Le fichier de données est vide.")
            if df.isnull().any().any():
                raise ValueError("Les données contiennent des valeurs manquantes.")
            return df
        except Exception as e:
            raise ValueError(f"Erreur de chargement des données : {str(e)}")

    def train(self, data_path: str) -> Dict[str, Any]:
        """Entraîne le modèle sur les données et retourne les résultats."""
        try:
            df = self.load_data(data_path)
            if len(df) < self.n_clusters:
                raise ValueError("Le nombre de lignes doit être >= au nombre de clusters.")

            X_scaled = self.scaler.fit_transform(df.values)

            self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
            labels = self.model.fit_predict(X_scaled)

            score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1

            self.results = {
                "model": "AgglomerativeClustering",
                "parameters": {
                    "n_clusters": self.n_clusters,
                    "linkage": self.linkage
                },
                "metrics": {"silhouette_score": score},
                "labels": labels.tolist(),
                "visualization_data": X_scaled[:, :2].tolist() if X_scaled.shape[1] >= 2 else X_scaled.tolist()
            }
            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="app/data/resultats/agglo_results.json") -> str:
        """Sauvegarde les résultats du clustering dans un fichier JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        return output_path


def run(X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Fonction utilitaire pour exécuter rapidement un clustering hiérarchique.

    Args:
        X       : Données d'entrée (DataFrame ou ndarray)
        kwargs  : n_clusters (int), linkage (str)

    Returns:
        dict contenant les labels, le score silhouette et les données de visualisation.
    """
    try:
        n_clusters = kwargs.get("n_clusters", 3)
        linkage = kwargs.get("linkage", "ward")

        if X.isnull().any().any():
            raise ValueError("Les données contiennent des valeurs manquantes.")

        scaler = StandardScaler()
        X_arr = X.values if hasattr(X, "values") else X
        X_scaled = scaler.fit_transform(X_arr)

        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1

        results = {
            "metrics": {"silhouette_score": score},
            "cluster_labels": labels.tolist(),
            "model": "AgglomerativeClustering",
            "parameters": {"n_clusters": n_clusters, "linkage": linkage},
            "visualization_data": X_scaled[:, :2].tolist() if X_scaled.shape[1] >= 2 else X_scaled.tolist()
        }

        return results

    except Exception as e:
        raise RuntimeError(f"Erreur complète Agglomerative.run : {str(e)}")
