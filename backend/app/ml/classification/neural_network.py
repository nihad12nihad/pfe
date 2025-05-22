import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
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
from typing import Dict, Any, Union, Optional

logging.basicConfig(level=logging.INFO)

class NeuralNetworkModel:
    """Modèle Réseau de Neurones (MLPClassifier) avec entraînement, prédiction et sauvegarde des résultats."""

    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.model: Optional[MLPClassifier] = None
        self.label_encoder = LabelEncoder()
        self.results: Dict[str, Any] = {}

    def load_data(self, data_path: str, target_column: str) -> tuple[pd.DataFrame, np.ndarray]:
        """Charge et valide les données utilisateur."""
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
            raise ValueError(f"Erreur de chargement des données : {str(e)}")

    def train(self, data_path: str, target_column: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Entraîne et évalue le modèle de réseau de neurones."""
        try:
            X, y = self.load_data(data_path, target_column)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            self.results = {
                "model": "NeuralNetwork",
                "parameters": {
                    "hidden_layer_sizes": self.hidden_layer_sizes,
                    "activation": self.activation,
                    "solver": self.solver,
                    "max_iter": self.max_iter,
                    "test_size": test_size,
                },
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

            logging.info("Modèle Réseau de Neurones entraîné avec succès.")
            return self.results

        except Exception as e:
            logging.error(f"Erreur pendant l'entraînement : {e}")
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path: str = "app/data/resultats/neural_network_results.json") -> str:
        """Sauvegarde les résultats au format JSON."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            logging.info(f"Résultats sauvegardés dans {output_path}.")
            return output_path
        except Exception as e:
            raise ValueError(f"Erreur lors de la sauvegarde : {str(e)}")

    def predict(self, X_new: Union[pd.DataFrame, np.ndarray]) -> list:
        """Effectue une prédiction sur de nouvelles données."""
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        try:
            predictions = self.model.predict(X_new)
            return self.label_encoder.inverse_transform(predictions).tolist()
        except Exception as e:
            raise ValueError(f"Erreur lors de la prédiction : {str(e)}")


def run(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Entraîne un MLPClassifier et retourne un dict standardisé.
    
    X : DataFrame des features  
    y : Series de la cible (objet ou numérique)  
    kwargs : paramètres additionnels pour MLPClassifier  
    """
    try:
        label_encoder = LabelEncoder()
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            y_enc = label_encoder.fit_transform(y)
            class_labels = list(label_encoder.classes_)
        else:
            y_enc = y.values
            class_labels = list(np.unique(y_enc))

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_enc, test_size=test_size, random_state=random_state
        )

        model = MLPClassifier(random_state=random_state, **kwargs)
        model.fit(X_train, y_train)
        y_pred_enc = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_enc),
            "precision": precision_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_enc, average="weighted", zero_division=0)
        }

        cm = confusion_matrix(y_test, y_pred_enc)

        if hasattr(label_encoder, "inverse_transform") and (y.dtype == 'object' or str(y.dtype).startswith('category')):
            y_pred = label_encoder.inverse_transform(y_pred_enc).tolist()
        else:
            y_pred = y_pred_enc.tolist()

        return {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "class_labels": class_labels,
            "visualization_data": X_test,
            "predictions": y_pred,
            "model": "MLPClassifier"
        }

    except Exception as e:
        raise RuntimeError(f"Erreur complète dans NeuralNetwork.run : {e}")
