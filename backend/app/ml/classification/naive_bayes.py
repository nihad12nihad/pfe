import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json
import logging
from typing import Dict, Any, Union

# Configuration des logs
logging.basicConfig(level=logging.INFO)

class NaiveBayesModel:
    """Modèle Naive Bayes avec détection automatique du type (gaussian/multinomial)."""

    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.model_type = "auto"

    def load_data(self, data_path: str, target_column: str) -> tuple[pd.DataFrame, np.ndarray]:
        try:
            df = pd.read_csv(data_path)

            if target_column not in df.columns:
                raise ValueError(f"Colonne cible '{target_column}' introuvable.")
            if df.empty or len(df) < 10:
                raise ValueError("Fichier vide ou insuffisant (min. 10 lignes).")

            X = df.drop(columns=[target_column])
            y = self.label_encoder.fit_transform(df[target_column])

            return X, y

        except Exception as e:
            logging.error(f"Erreur de chargement : {e}")
            raise

    def _choose_model_type(self, X: pd.DataFrame, model_type: str = "auto") -> str:
        if model_type != "auto":
            return model_type
        if all(np.issubdtype(dtype, np.integer) for dtype in X.dtypes):
            return "multinomial"
        elif all(np.issubdtype(dtype, np.floating) for dtype in X.dtypes):
            return "gaussian"
        raise ValueError("Colonnes avec types mixtes non supportées en mode auto.")

    def train(self, data_path: str, target_column: str, test_size: float = 0.2, model_type: str = "auto") -> Dict[str, Any]:
        try:
            X, y = self.load_data(data_path, target_column)
            self.model_type = self._choose_model_type(X, model_type)

            self.model = GaussianNB() if self.model_type == "gaussian" else MultinomialNB()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            self.results = {
                "model": f"NaiveBayes_{self.model_type}",
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

            logging.info(f"Modèle {self.model_type} entraîné avec succès.")
            return self.results

        except Exception as e:
            logging.error(f"Erreur pendant l'entraînement : {e}")
            return {"error": str(e), "status": "failed"}

    def predict(self, X_new: Union[pd.DataFrame, np.ndarray]) -> list:
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        try:
            predictions = self.model.predict(X_new)
            return self.label_encoder.inverse_transform(predictions).tolist()
        except Exception as e:
            raise ValueError(f"Erreur lors de la prédiction : {str(e)}")

    def save_results(self, output_path: str = "app/data/resultats/naive_bayes_results.json") -> str:
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            logging.info(f"Résultats sauvegardés dans {output_path}.")
            return output_path
        except Exception as e:
            raise ValueError(f"Erreur lors de la sauvegarde : {str(e)}")


def run(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    try:
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y)
        class_labels = list(label_encoder.classes_)

        if model_type == "auto":
            if all(np.issubdtype(dtype, np.integer) for dtype in X.dtypes):
                model_type = "multinomial"
            else:
                model_type = "gaussian"

        model = GaussianNB() if model_type == "gaussian" else MultinomialNB()

        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return {
            "metrics": {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            },
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "class_labels": class_labels,
            "visualization_data": X_test.to_numpy(),
            "predictions": label_encoder.inverse_transform(y_pred).tolist(),
            "model": f"NaiveBayes_{model_type}"
        }

    except Exception as e:
        raise RuntimeError(f"Erreur complète dans NaiveBayes.run : {e}")
