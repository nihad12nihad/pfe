# core/algorithms.py
from typing import Dict, Any
import pandas as pd
from pathlib import Path
import logging
import inspect
# Importez tous vos algorithmes comme avant
from app.ml.classification import knn, decision_tree, naive_bayes, neural_network, random_forest, svm
from app.ml.regression import linear_regression, multiple_regression
from app.ml.clustering import kmeans, dbscan, agglomerative
from app.ml.association import apriori

logger = logging.getLogger(__name__)

def get_algorithm(name: str):
    """Retourne la fonction de l'algorithme avec un wrapper standardisé"""
    algorithms = {
        "knn": standardized_wrapper(knn.run),
        "decision_tree": standardized_wrapper(decision_tree.run),
        "naive_bayes": standardized_wrapper(naive_bayes.run),
        "neural_network": standardized_wrapper(neural_network.run),
        "random_forest": standardized_wrapper(random_forest.run),
        "svm": standardized_wrapper(svm.run),
        "linear_regression": standardized_wrapper(linear_regression.run),
        "multiple_regression": standardized_wrapper(multiple_regression.run),
        "kmeans": standardized_wrapper(kmeans.run),
        "dbscan": standardized_wrapper(dbscan.run),
        "agglomerative": standardized_wrapper(agglomerative.run),
        "apriori": standardized_wrapper(apriori.run),
    }

    name = name.lower()
    if name in algorithms:
        return algorithms[name]
    else:
        raise ValueError(f"L'algorithme '{name}' n'est pas pris en charge.")

def standardized_wrapper(algo_func):
    """Wrapper pour standardiser la sortie de tous les algorithmes"""
    def wrapper(data_path: str, target_column: str = None, **params):
        try:
            # 1. Chargement des données
            logger.info(f"Chargement des données depuis {data_path}")
            data = pd.read_csv(data_path)

            # 2. Déterminer les paramètres de la fonction
            signature = inspect.signature(algo_func)
            accepts_y = 'y' in signature.parameters

            if target_column and accepts_y:
                if target_column not in data.columns:
                    raise ValueError(f"Colonne cible '{target_column}' non trouvée dans le fichier.")
                X = data.drop(columns=[target_column])
                y = data[target_column]
                raw_result = algo_func(X, y, **params)
            else:
                X = data.drop(columns=[target_column]) if target_column and target_column in data.columns else data
                raw_result = algo_func(X, **params)

            # 3. Standardisation des résultats
            logger.info("Standardisation des résultats")
            return standardize_output(raw_result, algo_func.__module__)

        except Exception as e:
            logger.error(f"Erreur dans l'algorithme: {str(e)}", exc_info=True)
            raise RuntimeError(f"Erreur d'exécution: {str(e)}")

    return wrapper

def standardize_output(raw_result: Dict[str, Any], algo_module: str) -> Dict[str, Any]:
    """Transforme les résultats bruts en format standard"""
    standardized = {
        'metrics': raw_result.get('metrics', {}),
        'model': raw_result.get('model'),
        'predictions': raw_result.get('predictions', []),
        'visualization_data': raw_result.get('visualization_data', {})
    }
    
    # Ajouts spécifiques selon le type d'algorithme
    if 'classification' in algo_module:
        standardized.update({
            'confusion_matrix': raw_result.get('confusion_matrix'),
            'class_labels': raw_result.get('class_labels', [])
        })
    elif 'regression' in algo_module:
        standardized.update({
            'coefficients': raw_result.get('coefficients', {}),
            'residuals': raw_result.get('residuals', [])
        })
    elif 'clustering' in algo_module:
        standardized.update({
            'cluster_labels': raw_result.get('cluster_labels', []),
            'centers': raw_result.get('centers', [])
        })
    elif 'association' in algo_module:
        standardized.update({
            'frequent_itemsets': raw_result.get('frequent_itemsets', {}),
            'association_rules': raw_result.get('association_rules', [])
        })
    
    return standardized
