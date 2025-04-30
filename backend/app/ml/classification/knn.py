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

    def save_results(self, output_path="results/knn_results.json"):
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

def run(data_path, target_column, n_neighbors=5, weights="uniform", algorithm="auto", test_size=0.2):
    knn = KNNModel(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    results = knn.train(data_path, target_column, test_size)
    return results
