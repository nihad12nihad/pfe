import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
from pathlib import Path
from typing import Dict, Any


class DBSCANClustering:
    """Classe pour appliquer DBSCAN Clustering avec évaluation et sauvegarde des résultats."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Charge le jeu de données depuis un fichier CSV."""
        try:
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError("Le fichier de données est vide")
            if df.isnull().any().any():
                raise ValueError("Les données contiennent des valeurs manquantes.")
            return df
        except Exception as e:
            raise ValueError(f"Erreur de chargement des données : {str(e)}")

    def train(self, data_path: str) -> Dict[str, Any]:
        """Entraîne le modèle DBSCAN sur les données fournies."""
        try:
            df = self.load_data(data_path)
            X_scaled = self.scaler.fit_transform(df)

            self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = self.model.fit_predict(X_scaled)

            unique_labels = set(labels)
            # Pour silhouette_score, on ignore les labels -1 (bruit)
            n_clusters = len(unique_labels - {-1})
            silhouette = silhouette_score(X_scaled, labels) if n_clusters > 1 else -1

            self.results = {
                "model": "DBSCAN",
                "parameters": {
                    "eps": self.eps,
                    "min_samples": self.min_samples
                },
                "metrics": {"silhouette_score": silhouette},
                "labels": labels.tolist(),
                "visualization_data": X_scaled[:, :2].tolist() if X_scaled.shape[1] >= 2 else X_scaled.tolist()
            }
            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="app/data/resultats/dbscan_results.json") -> str:
        """Sauvegarde les résultats dans un fichier JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        return output_path


def run(
    X: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    Applique DBSCAN sur X et renvoie un dict standardisé.

    Args:
        X           : DataFrame des features
        eps         : rayon de voisinage
        min_samples : nombre minimum de points dans eps-voisinage
        kwargs      : paramètres supplémentaires ignorés ici

    Returns:
        dict avec métriques, labels, données pour visualisation, etc.
    """
    try:
        if X.isnull().any().any():
            raise ValueError("Les données contiennent des valeurs manquantes.")

        scaler = StandardScaler()
        X_arr = X.values  # garantit un numpy.ndarray
        X_scaled = scaler.fit_transform(X_arr)

        model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = model.fit_predict(X_scaled)

        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        score = silhouette_score(X_scaled, labels) if n_clusters > 1 else -1

        viz = X_scaled[:, :2].tolist() if X_scaled.shape[1] >= 2 else X_scaled.tolist()

        return {
            "metrics": {"silhouette_score": score},
            "cluster_labels": labels.tolist(),
            "visualization_data": viz,
            "model": "DBSCAN",
            "parameters": {"eps": eps, "min_samples": min_samples}
        }

    except Exception as e:
        raise RuntimeError(f"Erreur complète DBSCAN.run : {e}")
