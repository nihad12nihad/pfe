import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)

class SVMModel:
    """Classe Support Vector Machine (SVM) avec gestion d'erreurs et sauvegarde des résultats."""

    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.label_encoder = LabelEncoder()
        self.results = {}

    def load_data(self, data_path, target_column):
        """Charge et valide les données utilisateur."""
        try:
            df = pd.read_csv(data_path)
            if target_column not in df.columns:
                raise ValueError(f"Colonne cible '{target_column}' introuvable.")
            if df.empty:
                raise ValueError("Le fichier de données est vide.")
            if len(df) < 10:
                raise ValueError("Trop peu d'échantillons (min 10 requis).")

            X = df.drop(columns=[target_column])
            y = self.label_encoder.fit_transform(df[target_column])
            return X, y

        except Exception as e:
            raise ValueError(f"Erreur de chargement des données : {str(e)}")

    def train(self, data_path, target_column, test_size=0.2):
        """Entraîne et évalue le modèle SVM."""
        try:
            X, y = self.load_data(data_path, target_column)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                random_state=42
            )
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            self.results = {
                "model": "SVM",
                "parameters": {
                    "kernel": self.kernel,
                    "C": self.C,
                    "gamma": self.gamma,
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
            logging.info("Modèle SVM entraîné avec succès.")
            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="results/svm_results.json"):
        """Sauvegarde des résultats au format JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Résultats sauvegardés dans {output_path}.")
        return output_path

def run(data_path, target_column, test_size=0.2):
    svm_model = SVMModel()
    results = svm_model.train(data_path, target_column, test_size)
    return results
