# core/algorithms.py

from ml.classification import knn, decision_tree, naive_bayes, neural_network, random_forest, svm
from ml.regression import linear_regression, multiple_regression
from ml.clustering import kmeans, dbscan, agglomerative
from ml.association import apriori

def get_algorithm(name: str):
    algorithms = {
        "knn": knn.run,
        "decision_tree": decision_tree.run,
        "naive_bayes": naive_bayes.run,
        "neural_network": neural_network.run,
        "random_forest": random_forest.run,
        "svm": svm.run,
        "linear_regression": linear_regression.run,
        "multiple_regression": multiple_regression.run,
        "kmeans": kmeans.run,
        "dbscan": dbscan.run,
        "agglomerative": agglomerative.run,
        "apriori": apriori.run,
    }

    name = name.lower()
    if name in algorithms:
        return algorithms[name]
    else:
        raise ValueError(f"L'algorithme '{name}' n'est pas pris en charge.")
