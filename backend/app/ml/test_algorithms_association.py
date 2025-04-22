import logging
from association.apriori import AprioriAssociation

logging.basicConfig(level=logging.INFO)

def test_apriori(data_path):
    """Test de l'algorithme Apriori."""
    try:
        model = AprioriAssociation(min_support=0.3, metric='lift', min_threshold=1.0)
        results = model.train(data_path)

        if "error" in results:
            logging.error(f"Erreur: {results['error']}")
        else:
            logging.info(f"Nombre total d'itemsets fréquents: {results['summary']['total_itemsets']}")
            logging.info(f"Nombre total de règles générées: {results['summary']['total_rules']}")
            model.save_results("results/apriori_results.json")
            logging.info("Résultats sauvegardés dans results/apriori_results.json")
    except Exception as e:
        logging.error(f"Erreur dans Apriori: {str(e)}")

if __name__ == "__main__":
    data_path = "datasets/market_basket.csv"  # Mets le bon chemin ici
    test_apriori(data_path)
