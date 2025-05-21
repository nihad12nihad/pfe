import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json
import logging
import pickle
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, Optional

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionTreeModel:
    def __init__(self, criterion: str = "gini", max_depth: Optional[int] = None, random_state: int = 42):
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, random_state=self.random_state)
        self.label_encoder = LabelEncoder()
        self.results: Dict[str, Any] = {}
        self.feature_importances_: Optional[np.ndarray] = None

    def load_data(self, data_path: Union[str, Path], target_column: str) -> tuple[pd.DataFrame, pd.Series]:
        try:
            df = pd.read_csv(data_path)

            if target_column not in df.columns:
                raise ValueError(f"Colonne cible '{target_column}' introuvable. Colonnes disponibles: {list(df.columns)}")
            if df.empty:
                raise ValueError("Le fichier de données est vide.")
            if len(df[target_column].unique()) < 2:
                raise ValueError("La colonne cible doit avoir au moins 2 classes.")
            if len(df) < 20:
                logger.warning("Dataset très petit (moins de 20 échantillons)")

            X = df.drop(columns=[target_column])
            y = self.label_encoder.fit_transform(df[target_column])

            return X, y

        except Exception as e:
            logger.error(f"Erreur de chargement des données: {str(e)}")
            raise

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        try:
            if y.dtype == 'object' or str(y.dtype).startswith('category'):
                y = self.label_encoder.fit_transform(y)
                class_labels = {i: label for i, label in enumerate(self.label_encoder.classes_)}
            else:
                class_labels = list(np.unique(y))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.feature_importances_ = self.model.feature_importances_

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }

            cm = confusion_matrix(y_test, y_pred)

            roc_data = {}
            if hasattr(self.model, "predict_proba"):
                try:
                    y_prob = self.model.predict_proba(X_test)
                    if y_prob.shape[1] == 2:
                        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
                        roc_data = {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "thresholds": thresholds.tolist(),
                            "auc": auc(fpr, tpr)
                        }
                except Exception as e:
                    logger.warning(f"Erreur lors du calcul de la courbe ROC: {e}")

            self.results = {
                'model_type': 'decision_tree',
                'parameters': {
                    'criterion': self.criterion,
                    'max_depth': self.max_depth,
                    'random_state': self.random_state,
                },
                'metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'feature_importances': dict(zip(X.columns, self.feature_importances_)),
                'class_labels': class_labels,
                'roc_curve': roc_data
            }

            self.save_tree_graph(X.columns)

            logger.info("Entraînement terminé avec succès")
            return self.results

        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
            raise RuntimeError(f"Échec de l'entraînement : {e}")

    def save_model(self, output_path: Union[str, Path] = None) -> str:
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Le modèle doit être entraîné avant sauvegarde")

        if output_path is None:
            output_path = Path(__file__).parent.parent / "data" / "models" / "decision_tree.pkl"
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'label_encoder': self.label_encoder,
                    'metadata': {
                        'criterion': self.criterion,
                        'max_depth': self.max_depth,
                        'features': list(self.results.get('feature_importances', {}).keys())
                    }
                }, f)
            logger.info(f"Modèle sauvegardé dans {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Échec de la sauvegarde du modèle: {str(e)}")
            raise IOError(f"Impossible de sauvegarder le modèle: {str(e)}")

    def save_results(self, output_path: Union[str, Path] = None) -> str:
        if not self.results:
            raise ValueError("Aucun résultat à sauvegarder")

        if output_path is None:
            output_path = Path(__file__).parent.parent / "data" / "resultats" / "decision_tree_results.json"
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False, default=str)
            logger.info(f"Résultats sauvegardés dans {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Échec de la sauvegarde des résultats: {str(e)}")
            raise IOError(f"Impossible de sauvegarder les résultats: {str(e)}")

    def save_tree_graph(self, feature_names: list[str], output_path: Union[str, Path] = None) -> str:
        if output_path is None:
            output_path = Path(__file__).parent.parent / "data" / "resultats" / "decision_tree_graph.png"
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            plt.figure(figsize=(20, 10))
            plot_tree(self.model, feature_names=feature_names, class_names=True, filled=True)
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Graphique de l’arbre sauvegardé dans {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Erreur de sauvegarde du graphe de l’arbre: {e}")
            return ""

    def predict(self, X_new: pd.DataFrame) -> list:
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        try:
            predictions = self.model.predict(X_new)
            return self.label_encoder.inverse_transform(predictions).tolist()
        except Exception as e:
            logger.error(f"Erreur de prédiction: {str(e)}")
            raise

# Fonction utilitaire pour usage fonctionnel
def run(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42, **kwargs) -> Dict[str, Any]:
    try:
        label_encoder = LabelEncoder()
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            y_enc = label_encoder.fit_transform(y)
            class_labels = list(label_encoder.classes_)
        else:
            y_enc = y.values
            class_labels = list(np.unique(y_enc))

        X_train, X_test, y_train, y_test = train_test_split(X.values, y_enc, test_size=test_size, random_state=random_state)
        model = DecisionTreeClassifier(random_state=random_state, **kwargs)
        model.fit(X_train, y_train)
        y_pred_enc = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_enc),
            "precision": precision_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_enc, average="weighted", zero_division=0)
        }

        cm = confusion_matrix(y_test, y_pred_enc)

        if hasattr(label_encoder, "inverse_transform"):
            y_pred = label_encoder.inverse_transform(y_pred_enc).tolist()
            y_test_decoded = label_encoder.inverse_transform(y_test).tolist()
        else:
            y_pred = y_pred_enc.tolist()
            y_test_decoded = y_test.tolist()

        return {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "class_labels": class_labels,
            "visualization_data": np.column_stack((X_test,)),
            "actual": y_test_decoded,
            "predictions": y_pred,
            "model": "DecisionTreeClassifier",
            "feature_importances": dict(zip(X.columns, model.feature_importances_)) if hasattr(model, "feature_importances_") else None
        }

    except Exception as e:
        raise RuntimeError(f"Erreur complète dans DecisionTree.run : {e}")
