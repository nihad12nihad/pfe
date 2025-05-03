import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path
import json
import numpy as np


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

    def train(self, df):
        """
        Entraîne l'algorithme Apriori.
        Args:
            df (pd.DataFrame): Données transactionnelles binaires.
        Returns:
            dict: Résultats avec itemsets fréquents et règles.
        """
        try:
            if df.empty:
                raise ValueError("Le DataFrame fourni est vide.")

            df = df.astype(bool)

            # Itemsets fréquents
            frequent_itemsets = apriori(df, min_support=self.min_support, use_colnames=True)

            # Règles d'association
            rules = association_rules(frequent_itemsets, metric=self.metric, min_threshold=self.min_threshold)
            rules["antecedents"] = rules["antecedents"].apply(list)
            rules["consequents"] = rules["consequents"].apply(list)

            # Résumé + conversion
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

    def save_results(self, output_path="app/data/resultats/apriori_results.json"):
        """Sauvegarde les résultats au format JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        return output_path


def run(data_path, min_support=0.5, metric='lift', min_threshold=1.0):
    """
    Fonction d'exécution pour l'algorithme Apriori.
    Args:
        data_path (str | Path | pd.DataFrame): Chemin vers un CSV OU DataFrame déjà chargé.
    Returns:
        dict: Résultat formaté comme les autres algorithmes de la plateforme
    """
    try:
        # Charger les données
        if isinstance(data_path, (str, Path)):
            if not Path(data_path).is_file():
                raise FileNotFoundError(f"Le fichier '{data_path}' n'existe pas ou est inaccessible.")
            df = pd.read_csv(data_path)
        elif isinstance(data_path, pd.DataFrame):
            df = data_path
        else:
            raise TypeError("data_path doit être un chemin vers un fichier CSV ou un DataFrame")

        # Instancier et entraîner le modèle
        model = AprioriAssociation(min_support=min_support, metric=metric, min_threshold=min_threshold)
        result = model.train(df)

        if "error" in result:
            raise ValueError(result["error"])

        # Sortie harmonisée
        return {
            "metrics": {
                "total_itemsets": result["summary"]["total_itemsets"],
                "total_rules": result["summary"]["total_rules"]
            },
            "confusion_matrix": [],
            "class_labels": [],
            "visualization_data": df.values.tolist(),  # visualisation possible
            "predictions": [],
            "rules": result["association_rules"],
            "frequent_itemsets": result["frequent_itemsets"],
            "model": "Apriori"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "metrics": {},
            "confusion_matrix": [],
            "class_labels": [],
            "visualization_data": [],
            "predictions": [],
            "model": "Apriori"
        }
