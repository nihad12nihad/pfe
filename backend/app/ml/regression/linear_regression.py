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
    def __init__(self):
        self.model = LinearRegression()
        self.results = {}

    def load_data(self, data_path, target_column):
        """Charge les données et vérifie qu'il s'agit d'une seule variable explicative."""
        df = pd.read_csv(data_path)
        if target_column not in df.columns:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée dans les données.")

        X = df.drop(columns=[target_column])

        if X.shape[1] != 1:
            raise ValueError("Régression linéaire simple requiert une seule variable explicative.")

        y = df[target_column]
        return X, y

    def train(self, data_path, target_column, test_size=0.2):
        """Entraîne et évalue une régression linéaire simple."""
        try:
            X, y = self.load_data(data_path, target_column)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            # Supprimer les lignes avec NaN pour éviter les erreurs
            test_df = pd.concat([X_test, y_test], axis=1).dropna()
            X_test = test_df[X.columns]
            y_test = test_df[y.name]
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)

            self.results = {
                "model": "Linear Regression Simple",
                "coefficients": self.model.coef_.tolist(),
                "intercept": self.model.intercept_,
                "metrics": {
                    "mean_squared_error": mean_squared_error(y_test, predictions),
                    "mean_absolute_error": mean_absolute_error(y_test, predictions),
                    "r2_score": r2_score(y_test, predictions),
                }
            }

            logging.info("Modèle Linear Regression Simple entraîné avec succès.")
            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="app/data/resultats/linear_regression_results.json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Résultats sauvegardés dans {output_path}.")
        return output_path

def run(X: pd.DataFrame, y: pd.Series, test_size=0.2) -> dict:
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split

        # Supprimer les lignes contenant des valeurs manquantes
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
                "r2_score": r2_score(y_test, predictions)
            },
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_,
            "visualization_data": X_test.to_numpy().tolist(),
            "predictions": predictions.tolist(),
            "model": "LinearRegression"
        }
    except Exception as e:
        raise RuntimeError(f"Erreur complète LinearRegression: {str(e)}")


