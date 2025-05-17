import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)

class MultipleRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.results: Dict[str, Any] = {}

    def load_data(self, data_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
        """Charge les données pour une régression multiple."""
        df = pd.read_csv(data_path)
        if target_column not in df.columns:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée dans les données.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        if X.shape[1] < 2:
            raise ValueError("Régression multiple nécessite au moins deux variables explicatives.")
        return X, y

    def train(self, data_path: str, target_column: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Entraîne et évalue une régression linéaire multiple."""
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
            y_pred = self.model.predict(X_test)

            # Construire le résultat
            self.results = {
                "model": "Multiple Linear Regression",
                "parameters": {
                    "test_size": test_size
                },
                "coefficients": dict(zip(X.columns, self.model.coef_)),
                "intercept": self.model.intercept_,
                "metrics": {
                    "mean_squared_error": mean_squared_error(y_test, y_pred),
                    "mean_absolute_error": mean_absolute_error(y_test, y_pred),
                    "r2_score": r2_score(y_test, y_pred),
                },
                "predictions": y_pred.tolist(),      # pour la visualisation réel vs prédit
                "actual": y_test.tolist(),           # valeurs réelles
                "visualization_data": (              # pour scatter ou clusters
                    X_test.iloc[:, :2].values.tolist()
                    if X_test.shape[1] >= 2 else
                    X_test.values.tolist()
                )
            }

            logging.info("Modèle Multiple Regression entraîné avec succès.")
            return self.results

        except Exception as e:
            logging.error(f"Erreur lors de l'entraînement Multiple Regression: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path: str = "app/data/resultats/multiple_regression_results.json") -> str:
        """Sauvegarde les résultats au format JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        logging.info(f"Résultats sauvegardés dans {output_path}.")
        return output_path

def run(X: pd.DataFrame, y: pd.Series, test_size=0.2) -> Dict[str, Any]:
    try:
        # Supprimer les lignes avec valeurs manquantes
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
            "coefficients": dict(zip(X.columns, model.coef_)),
            "intercept": model.intercept_,
            "visualization_data": X_test.to_numpy().tolist(),
            "predictions": predictions.tolist(),
            "model": "MultipleLinearRegression"
        }
    except Exception as e:
        raise RuntimeError(f"Erreur complète MultipleRegression: {str(e)}")
