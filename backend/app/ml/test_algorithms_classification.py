from classification.knn import KNNModel
from classification.decision_tree import DecisionTreeModel
from classification.naive_bayes import NaiveBayesModel
from classification.neural_network import NeuralNetworkModel
from classification.random_forest import RandomForestModel
from classification.svm import SVMModel
import logging

logging.basicConfig(level=logging.INFO)

def test_algorithm(algorithm, data_path, target_column):
    """Test d'un algorithme de classification."""
    try:
        logging.info(f"Entraînement du modèle {algorithm.__class__.__name__}...")
        result = algorithm.train(data_path, target_column)
        if "error" in result:
            logging.error(f"Erreur dans le modèle {algorithm.__class__.__name__}: {result['error']}")
        else:
            logging.info(f"Résultats du modèle {algorithm.__class__.__name__}:")
            for metric, value in result["metrics"].items():
                logging.info(f"{metric.capitalize()}: {value:.4f}")
            logging.info(f"Matrice de confusion: {result['confusion_matrix']}")
            logging.info(f"Labels: {result['class_labels']}")
            algorithm.save_results(f"results/{algorithm.__class__.__name__.lower()}_results.json")
            logging.info(f"Résultats sauvegardés dans results/{algorithm.__class__.__name__.lower()}_results.json")
    except Exception as e:
        logging.error(f"Erreur dans le test de l'algorithme {algorithm.__class__.__name__}: {str(e)}")

def main():
    # Spécifie le chemin vers le dataset et la colonne cible
    data_path = "datasets/iris.csv"  # Remplace par le chemin réel de ton dataset
    target_column = "target"      # Remplace par le nom de ta colonne cible

    algorithms = [
        KNNModel(),
        DecisionTreeModel(),
        NaiveBayesModel(),
        NeuralNetworkModel(),
        RandomForestModel(),
        SVMModel()
    ]

    # Teste chaque algorithme
    for algo in algorithms:
        test_algorithm(algo, data_path, target_column)

if __name__ == "__main__":
    main()

