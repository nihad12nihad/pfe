import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import logging

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

    def save_results(self, output_path="results/linear_regression_results.json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Résultats sauvegardés dans {output_path}.")
        return output_path

def run(data_path, target_column, test_size=0.2):
    model = LinearRegressionModel()
    results = model.train(data_path, target_column, test_size=test_size)
    return results