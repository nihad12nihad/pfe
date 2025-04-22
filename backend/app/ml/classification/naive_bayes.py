import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
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

class NaiveBayesModel:
    """Classe Naive Bayes avec sélection automatique du type."""

    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.results = {}

    def load_data(self, data_path, target_column):
        """Charge les données et encode les labels."""
        try:
            df = pd.read_csv(data_path)
            if target_column not in df.columns:
                raise ValueError(f"Colonne cible '{target_column}' introuvable.")
            if df.empty:
                raise ValueError("Le fichier est vide.")
            if len(df) < 10:
                raise ValueError("Trop peu d'échantillons (min 10 requis).")

            X = df.drop(columns=[target_column])
            y = self.label_encoder.fit_transform(df[target_column])
            return X, y

        except Exception as e:
            raise ValueError(f"Erreur de chargement : {str(e)}")

    def train(self, data_path, target_column, test_size=0.2, model_type="auto"):
        """Entraîne le modèle Naive Bayes avec type auto ou défini."""
        try:
            X, y = self.load_data(data_path, target_column)

            # Détection automatique du type
            if model_type == "auto":
                if all(np.issubdtype(dtype, np.integer) for dtype in X.dtypes):
                    model_type = "multinomial"
                elif all(np.issubdtype(dtype, np.floating) for dtype in X.dtypes):
                    model_type = "gaussian"
                else:
                    raise ValueError("Types de données mixtes non pris en charge automatiquement.")

            # Choix du modèle
            if model_type == "gaussian":
                self.model = GaussianNB()
            elif model_type == "multinomial":
                self.model = MultinomialNB()
            else:
                raise ValueError("Type de modèle Naive Bayes invalide.")

            # Split, train, predict
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            self.results = {
                "model": f"NaiveBayes_{model_type}",
                "parameters": {"test_size": test_size},
                "metrics": {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                },
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "class_labels": {
                    int(i): label for i, label in enumerate(self.label_encoder.classes_)
                },
            }
            logging.info(f"Modèle {model_type} entraîné avec succès.")
            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="results/naive_bayes_results.json"):
        """Sauvegarde les résultats."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Résultats sauvegardés dans {output_path}.")
        return output_path

    def predict(self, X_new):
        """Prédit des classes sur de nouvelles données."""
        if self.model is None:
            raise ValueError("Modèle non entraîné.")
        try:
            predictions = self.model.predict(X_new)
            return self.label_encoder.inverse_transform(predictions).tolist()
        except Exception as e:
            raise ValueError(f"Erreur de prédiction : {str(e)}")

def run(data_path, target_column, test_size=0.2, model_type="auto"):
    nb_model = NaiveBayesModel()
    results = nb_model.train(data_path, target_column, test_size, model_type)
    return results
