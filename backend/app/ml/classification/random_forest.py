import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

class RandomForestModel:
    """Classe Random Forest avec gestion d'erreurs et sauvegarde des résultats."""

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
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
        """Entraîne et évalue le modèle Random Forest."""
        try:
            X, y = self.load_data(data_path, target_column)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            self.results = {
                "model": "RandomForest",
                "parameters": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
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
            logging.info("Modèle Random Forest entraîné avec succès.")
            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="app/data/resultats/random_forest_results.json"):
        """Sauvegarde des résultats au format JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Résultats sauvegardés dans {output_path}.")
        return output_path

def run(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Entraîne un RandomForestClassifier et retourne un dict standardisé.

    X : DataFrame des features  
    y : Series de la cible (objet ou numérique)  
    kwargs : paramètres additionnels pour RandomForestClassifier  
    """
    try:
        # 1) Encodage de y sans modifier y original
        label_encoder = LabelEncoder()
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            y_enc = label_encoder.fit_transform(y)
            class_labels = list(label_encoder.classes_)
        else:
            y_enc = y.values
            class_labels = list(np.unique(y_enc))

        # 2) Conversion de X en array et split
        X_arr = X.values
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_enc, test_size=test_size, random_state=random_state
        )

        # 3) Création et entraînement du modèle
        model = RandomForestClassifier(random_state=random_state, **kwargs)
        model.fit(X_train, y_train)

        # 4) Prédiction encodée
        y_pred_enc = model.predict(X_test)

        # 5) Calcul des métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_enc),
            "precision": precision_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_enc, average="weighted", zero_division=0)
        }

        # 6) Matrice de confusion
        cm = confusion_matrix(y_test, y_pred_enc)

        # 7) Restaurer les prédictions dans l’espace des labels d’origine
        if hasattr(label_encoder, "inverse_transform") and (y.dtype == 'object' or str(y.dtype).startswith('category')):
            y_pred = label_encoder.inverse_transform(y_pred_enc).tolist()
        else:
            y_pred = y_pred_enc.tolist()

        return {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "class_labels": class_labels,
            "visualization_data": X_test,    # np.ndarray pour scatter/clustering
            "predictions": y_pred,
            "model": "RandomForestClassifier"
        }

    except Exception as e:
        raise RuntimeError(f"Erreur complète dans RandomForest.run : {e}")