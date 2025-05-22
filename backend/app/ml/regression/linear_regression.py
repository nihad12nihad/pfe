import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

class LinearRegressionModel:
    """Régression linéaire simple avec gestion des erreurs et export des résultats."""

    def __init__(self):
        self.model = LinearRegression()
        self.results = {}

    def load_data(self, data_path: str, target_column: str):
        """Charge et valide les données pour une régression linéaire simple."""
        df = pd.read_csv(data_path)

        if target_column not in df.columns:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée dans les données.")
        
        if df.empty or df.shape[0] < 10:
            raise ValueError("Le fichier est vide ou contient moins de 10 lignes.")

        X = df.drop(columns=[target_column])
        if X.shape[1] != 1:
            raise ValueError("Régression linéaire simple requiert une seule variable explicative.")

        y = df[target_column]
        return X, y

    def train(self, data_path: str, target_column: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Entraîne le modèle de régression linéaire et retourne les métriques."""
        try:
            X, y = self.load_data(data_path, target_column)

            # Supprime les valeurs manquantes
            df_clean = pd.concat([X, y], axis=1).dropna()
            X = df_clean[X.columns]
            y = df_clean[y.name]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)

            self.results = {
                "model": "LinearRegression",
                "coefficients": self.model.coef_.tolist(),
                "intercept": float(self.model.intercept_),
                "metrics": {
                    "mean_squared_error": mean_squared_error(y_test, predictions),
                    "mean_absolute_error": mean_absolute_error(y_test, predictions),
                    "r2_score": r2_score(y_test, predictions),
                },
                "visualization_data": X_test.to_numpy().tolist(),
                "predictions": predictions.tolist()
            }

            logging.info("Modèle Linear Regression Simple entraîné avec succès.")
            return self.results

        except Exception as e:
            error_msg = f"Erreur dans l'entraînement du modèle : {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg, "status": "failed"}

    def save_results(self, output_path: str = "app/data/resultats/linear_regression_results.json") -> str:
        """Sauvegarde les résultats dans un fichier JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Résultats sauvegardés dans {output_path}.")
        return output_path


def run(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
    """Fonction exécutable pour entraînement rapide."""
    try:
        df = pd.concat([X, y], axis=1).dropna()
        X_clean = df[X.columns]
        y_clean = df[y.name]

        model = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        return {
            "metrics": {
                "mean_squared_error": mean_squared_error(y_test, predictions),
                "mean_absolute_error": mean_absolute_error(y_test, predictions),
                "r2_score": r2_score(y_test, predictions),
            },
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "visualization_data": X_test.to_numpy().tolist(),
            "predictions": predictions.tolist(),
            "model": "LinearRegression"
        }

    except Exception as e:
        raise RuntimeError(f"Erreur complète LinearRegression: {str(e)}")
