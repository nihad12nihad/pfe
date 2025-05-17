import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json
import logging
import pickle
from typing import Dict, Any, Union, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionTreeModel:
    """Classe Decision Tree avec gestion améliorée des erreurs et résultats standardisés."""

    def __init__(self, criterion: str = "gini", max_depth: Optional[int] = None, random_state: int = 42):
        """
        Initialise le modèle Decision Tree.
        
        Args:
            criterion: Mesure de qualité de split ("gini" ou "entropy")
            max_depth: Profondeur maximale de l'arbre
            random_state: Seed pour la reproductibilité
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.label_encoder = LabelEncoder()
        self.results: Dict[str, Any] = {}
        self.feature_importances_: Optional[np.ndarray] = None


    def load_data(self, data_path: Union[str, Path], target_column: str) -> tuple[pd.DataFrame, pd.Series]:
        """
        Charge et valide les données.
        
        Args:
            data_path: Chemin vers le fichier CSV
            target_column: Nom de la colonne cible
            
        Returns:
        Tuple (features, target)
            
        Raises:
            ValueError: Si les données sont invalides
        """
        try:
            df = pd.read_csv(data_path)
            
            # Validation des données
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
            # 1) Encodage conditionnel de y
            if y.dtype == 'object' or str(y.dtype).startswith('category'):
               y = self.label_encoder.fit_transform(y)
               class_labels = {i: label for i, label in enumerate(self.label_encoder.classes_)}
            else:
               class_labels = list(np.unique(y))

            # 2) Séparation train / test
            X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
            )

            # 3) Entraînement
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.feature_importances_ = np.array(self.feature_importances_)


            # 4) Calcul des métriques
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
            }
            y_test = np.array(y_test)  # Si y_test est une liste
            y_pred = np.array(y_pred)  # Si y_pred est une liste

            # 5) Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)

            # 6) Construction du résultat
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
                'class_labels': class_labels
            } 

            logger.info("Entraînement terminé avec succès")
            return self.results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
            raise RuntimeError(f"Échec de l'entraînement : {e}")


    def save_model(self, output_path: Union[str, Path] = None) -> str:
        """
        Sauvegarde le modèle entraîné de manière robuste.
        
        Args:
            output_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Chemin absolu où le modèle a été sauvegardé
            
        Raises:
            ValueError: Si le modèle n'est pas entraîné
            IOError: Si la sauvegarde échoue
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Le modèle doit être entraîné avant sauvegarde")

        # Chemin par défaut relatif au fichier courant
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
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Modèle sauvegardé avec succès dans {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Échec de la sauvegarde du modèle: {str(e)}")
            raise IOError(f"Impossible de sauvegarder le modèle: {str(e)}")

    def save_results(self, output_path: Union[str, Path] = None) -> str:
        """
        Sauvegarde les résultats au format JSON de manière robuste.
        
        Args:
            output_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Chemin absolu où les résultats ont été sauvegardés
            
        Raises:
            ValueError: Si aucun résultat n'est disponible
            IOError: Si la sauvegarde échoue
        """
        if not self.results:
            raise ValueError("Aucun résultat à sauvegarder")

        # Chemin par défaut relatif au fichier courant
        if output_path is None:
            output_path = Path(__file__).parent.parent / "data" / "resultats" / "decision_tree_results.json"
        
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, 
                         indent=4, 
                         ensure_ascii=False,
                         default=str)  # Pour sérialiser les types non JSON
            
            logger.info(f"Résultats sauvegardés avec succès dans {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Échec de la sauvegarde des résultats: {str(e)}")
            raise IOError(f"Impossible de sauvegarder les résultats: {str(e)}")

    def predict(self, X_new: pd.DataFrame) -> list:
        """
        Effectue des prédictions sur de nouvelles données.
        
        Args:
            X_new: DataFrame contenant les nouvelles données
            
        Returns:
            Liste des prédictions
            
        Raises:
            ValueError: Si le modèle n'est pas entraîné
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        try:
            predictions = self.model.predict(X_new)
            return self.label_encoder.inverse_transform(predictions).tolist()
        except Exception as e:
            logger.error(f"Erreur de prédiction: {str(e)}")
            raise

def run(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Entraîne un DecisionTreeClassifier et renvoie un dict standardisé.
    X : DataFrame des features
    y : Series de la cible (objet ou numérique)
    kwargs : tous les paramètres de DecisionTreeClassifier
    """
    try:
        # 1) Gérez l'encodage de y sans modifier y d'origine
        label_encoder = LabelEncoder()
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            y_enc = label_encoder.fit_transform(y)
            class_labels = list(label_encoder.classes_)
        else:
            y_enc = y.values
            class_labels = list(np.unique(y_enc))

        # 2) Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_enc, test_size=test_size, random_state=random_state
        )

        # 3) Création et entraînement du modèle
        model = DecisionTreeClassifier(random_state=random_state, **kwargs)
        model.fit(X_train, y_train)

        # 4) Prédictions
        y_pred_enc = model.predict(X_test)

        # 5) Calcul des métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_enc),
            "precision": precision_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_enc, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_enc, average="weighted", zero_division=0)
        }

        # 6) Matrice de confusion
        cm = confusion_matrix(y_test, y_pred_enc)

        # 7) Restaurer les prédictions dans l’espace des labels d’origine
        if hasattr(label_encoder, "inverse_transform") and (y.dtype == 'object' or str(y.dtype).startswith('category')):
            y_pred = label_encoder.inverse_transform(y_pred_enc).tolist()
        else:
            y_pred = y_pred_enc.tolist()

        return {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "class_labels": class_labels,
            "visualization_data": np.column_stack((X_test,)),  # X_test déjà array
            "predictions": y_pred,
            "model": "DecisionTreeClassifier"
        }

    except Exception as e:
        # remonte une erreur claire pour FastAPI
        raise RuntimeError(f"Erreur complète dans DecisionTree.run : {e}")
