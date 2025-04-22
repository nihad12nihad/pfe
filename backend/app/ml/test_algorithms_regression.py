import logging
from regression.linear_regression import LinearRegressionModel
from regression.multiple_regression import MultipleRegressionModel

logging.basicConfig(level=logging.INFO)

def test_regression_algorithm(algorithm, data_path, target_column):
    """Test d'un algorithme de régression."""
    try:
        logging.info(f"Entraînement du modèle {algorithm.__class__.__name__}...")
        result = algorithm.train(data_path, target_column)

        if "error" in result:
            logging.error(f"Erreur dans le modèle {algorithm.__class__.__name__}: {result['error']}")
        else:
            logging.info(f"Résultats du modèle {algorithm.__class__.__name__}:")
            logging.info(f"Intercept: {result['intercept']}")
            if isinstance(result["coefficients"], dict):
                for feature, coef in result["coefficients"].items():
                    logging.info(f"Coefficient {feature}: {coef:.4f}")
            else:
                logging.info(f"Coefficient: {result['coefficients'][0]:.4f}")

            for metric, value in result["metrics"].items():
                logging.info(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")

            algorithm.save_results(f"results/{algorithm.__class__.__name__.lower()}_results.json")
            logging.info(f"Résultats sauvegardés dans results/{algorithm.__class__.__name__.lower()}_results.json")

    except Exception as e:
        logging.error(f"Erreur lors du test de {algorithm.__class__.__name__}: {str(e)}")


def main():
    # Remplace par ton vrai dataset et le nom de la colonne cible
    data_path = "datasets/california_housing.csv"
    target_column = "MedHouseVal"

    algorithms = [
        LinearRegressionModel(),
        MultipleRegressionModel()
    ]

    for algo in algorithms:
        test_regression_algorithm(algo, data_path, target_column)

if __name__ == "__main__":
    main()
