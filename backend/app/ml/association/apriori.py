import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path
import json

class AprioriAssociation:
    """Classe pour l'algorithme Apriori — Association Rules Mining"""

    def __init__(self, min_support=0.5, metric='lift', min_threshold=1.0):
        self.min_support = min_support
        self.metric = metric
        self.min_threshold = min_threshold
        self.results = {}

    @staticmethod
    def convert_frozensets(data):
        """Convertit récursivement les frozensets en listes dans un dict/list."""
        if isinstance(data, list):
            return [AprioriAssociation.convert_frozensets(d) for d in data]
        elif isinstance(data, dict):
            return {k: AprioriAssociation.convert_frozensets(v) for k, v in data.items()}
        elif isinstance(data, frozenset):
            return list(data)
        else:
            return data

    def train(self, data_path):
        """
        Entraîne l'algorithme Apriori.
        Args:
            data_path (str): Chemin vers un fichier CSV transactionnel encodé (0/1 ou booléen).
        Returns:
            dict: Résultats avec itemsets fréquents et règles.
        """
        try:
            df = pd.read_csv(data_path)

            if df.empty:
                raise ValueError("Le fichier CSV est vide.")

            # Convertir les colonnes en booléens (obligatoire pour mlxtend)
            for col in df.columns:
                df[col] = df[col].astype(bool)

            # Générer les itemsets fréquents
            frequent_itemsets = apriori(df, min_support=self.min_support, use_colnames=True)

            # Générer les règles d'association
            rules = association_rules(frequent_itemsets, metric=self.metric, min_threshold=self.min_threshold)

            # Convertir les frozenset en listes pour le JSON
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x))

            # Remplir self.results
            self.results = {
                "frequent_itemsets": self.convert_frozensets(frequent_itemsets.to_dict(orient="records")),
                "association_rules": self.convert_frozensets(rules.to_dict(orient="records")),
                "summary": {
                    "total_itemsets": len(frequent_itemsets),
                    "total_rules": len(rules)
                }
            }

            return self.results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def save_results(self, output_path="results/apriori_results.json"):
        """Sauvegarde les résultats au format JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        return output_path

def run(data_path, min_support=0.5, metric='lift', min_threshold=1.0):
    model = AprioriAssociation(min_support=min_support, metric=metric, min_threshold=min_threshold)
    results = model.train(data_path)
    return results
