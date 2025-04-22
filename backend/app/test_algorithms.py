from core.algorithms import get_algorithm

def test_get_algorithm():
    # Tester les algorithmes connus
    known_algos = [
        "knn", "decision_tree", "naive_bayes", "neural_network",
        "random_forest", "svm", "linear_regression", "multiple_regression",
        "kmeans", "dbscan", "agglomerative", "apriori"
    ]

    for algo in known_algos:
        func = get_algorithm(algo)
        assert callable(func), f"{algo} n'a pas renvoyé une fonction valide."

    # Tester une erreur pour algo inconnu
    try:
        get_algorithm("invalide_algo")
    except ValueError as e:
        assert str(e) == "L'algorithme 'invalide_algo' n'est pas pris en charge."
    else:
        assert False, "Une exception aurait dû être levée pour un algorithme inconnu."
