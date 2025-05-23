import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import json
import logging
from typing import Dict, Any, Union, Optional

# Configuration de base des logs
logging.basicConfig(
    level=logging.INFO,
)

class KNNModel:
    """Classe implémentant l'algorithme KNN avec gestion complète des erreurs, normalisation et sortie standardisée"""

    def __init__(self, n_neighbors=5, weights="uniform", algorithm="auto"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.results = {}
        self.X_columns = None

    def load_data(self, data_path, target_column):
        """Charge et valide les données"""
        try:
            df = pd.read_csv(data_path)

            if target_column not in df.columns:
                raise ValueError(f"Colonne cible '{target_column}' introuvable.")
            if df.empty:
                raise ValueError("Le fichier de données est vide.")
            if len(df) < 10:
                raise ValueError("Trop peu d'échantillons (minimum 10 requis).")

            X = df.drop(columns=[target_column])
            y = self.label_encoder.fit_transform(df[target_column])

            # Vérification des colonnes numériques
            if not np.all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
                raise ValueError("Toutes les colonnes de caractéristiques doivent être numériques.")

            return X, y

        except Exception as e:
            logging.error(f"Erreur de chargement des données : {str(e)}")
            raise ValueError(f"Erreur de chargement des données : {str(e)}")

    def train(self, data_path, target_column, test_size=0.2):
        """Entraîne et évalue le modèle KNN"""
        try:
            X, y = self.load_data(data_path, target_column)
        
            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Entraînement
            self.model = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                algorithm=self.algorithm,
            )
            self.model.fit(X_train, y_train)

            # Prédictions
            y_pred = self.model.predict(X_test)

            # Résultats
            self.results = {
                "model": "KNN",
                "parameters": {
                    "n_neighbors": self.n_neighbors,
                    "weights": self.weights,
                    "algorithm": self.algorithm,
                    "test_size": test_size,
                },
                "metrics": {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted"),
                    "recall": recall_score(y_test, y_pred, average="weighted"),
                    "f1": f1_score(y_test, y_pred, average="weighted"),
                },
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "class_labels": {
                    int(i): label for i, label in enumerate(self.label_encoder.classes_)
                },
            }

            logging.info("Modèle entraîné avec succès.")
            return self.results

        except Exception as e:
            logging.error(f"Erreur pendant l'entraînement : {str(e)}")
            return {"error": str(e), "status": "failed"}

    def predict(self, X_new):
       
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")
        try:
            predictions = self.model.predict(X_new)
            return self.label_encoder.inverse_transform(predictions).tolist()
        
        except Exception as e:
            raise ValueError(f"Erreur lors de la prédiction : {str(e)}")

        except Exception as e:
            logging.error(f"Erreur de prédiction : {str(e)}")
            raise ValueError(f"Erreur de prédiction : {str(e)}")

    def save_results(self, output_path="app/data/resultats/knn_results.json"):
        """Sauvegarde les résultats au format JSON"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            logging.info(f"Résultats sauvegardés dans {output_path}.")
            return output_path
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des résultats : {str(e)}")
            raise ValueError(f"Erreur lors de la sauvegarde des résultats : {str(e)}")

def run(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Entraîne un KNeighborsClassifier et renvoie un dict standardisé.
    X : DataFrame des features
    y : Series de la cible (objet ou numérique)
    kwargs : paramètres de KNeighborsClassifier (n_neighbors, weights, algorithm, etc.)
    """
    try:
        # Encodage de y sans modifier l’original
        label_encoder = LabelEncoder()
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            y_enc = label_encoder.fit_transform(y)
            class_labels = list(label_encoder.classes_)
        else:
            y_enc = y.values
            class_labels = list(np.unique(y_enc))

        # Split train/test
        X_arr = X.values
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_enc, test_size=test_size, random_state=random_state
        )

        # Création et entraînement du modèle
        model = KNeighborsClassifier(**kwargs)
        model.fit(X_train, y_train)

        # Prédictions encodées
        y_pred_enc = model.predict(X_test)

        # Calcul des métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_enc),
            "precision": precision_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_enc, average="weighted", zero_division=0)
        }

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred_enc)

        # Restaurer les prédictions dans l’espace des labels d’origine
        if hasattr(label_encoder, "inverse_transform") and (y.dtype == 'object' or str(y.dtype).startswith('category')):
            y_pred = label_encoder.inverse_transform(y_pred_enc).tolist()
        else:
            y_pred = y_pred_enc.tolist()

        return {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "class_labels": class_labels,
            "visualization_data": X_test,          # déjà un np.ndarray
            "predictions": y_pred,
            "model": "KNeighborsClassifier"
        }

    except Exception as e:
        raise RuntimeError(f"Erreur complète dans KNN.run : {e}")