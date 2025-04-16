import logging
from clustering.kmeans import KMeansClustering
from clustering.dbscan import DBSCANClustering
from clustering.agglomerative import AgglomerativeClusteringModel

logging.basicConfig(level=logging.INFO)

def test_clustering(algorithm, data_path):
    """Test d'un algorithme de clustering."""
    try:
        logging.info(f"Entraînement du modèle {algorithm.__class__.__name__}...")
        result = algorithm.train(data_path)

        if "error" in result:
            logging.error(f"Erreur pour {algorithm.__class__.__name__}: {result['error']}")
        else:
            logging.info(f"Résultats du modèle {algorithm.__class__.__name__}:")
            logging.info(f"Silhouette Score: {result['silhouette_score']:.4f}")
            logging.info(f"Nombre de clusters trouvés: {len(set(result['labels']))}")
            algorithm.save_results(f"results/{algorithm.__class__.__name__.lower()}_results.json")
            logging.info(f"Résultats sauvegardés dans results/{algorithm.__class__.__name__.lower()}_results.json")

    except Exception as e:
        logging.error(f"Erreur dans le test de {algorithm.__class__.__name__}: {str(e)}")

def main():
    # Spécifie le chemin vers le dataset SANS colonne cible
    data_path = "chemin/vers/ton_dataset.csv"  # Remplace par ton vrai chemin !

    algorithms = [
        KMeansClustering(n_clusters=3),
        DBSCANClustering(eps=0.5, min_samples=5),
        AgglomerativeClusteringModel(n_clusters=3, linkage="ward")
    ]

    for algo in algorithms:
        test_clustering(algo, data_path)

if __name__ == "__main__":
    main()
