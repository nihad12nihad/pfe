import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def handle_missing_values(df, numeric_strategy='mean', categorical_strategy='constant'):
    """
    Gère les valeurs manquantes dans un DataFrame.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy=numeric_strategy)
        df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]),
                                        columns=numeric_cols, index=df.index)

    if len(categorical_cols) > 0:
        if categorical_strategy == 'constant':
            df[categorical_cols] = df[categorical_cols].fillna('MANQUANT')
        else:
            imputer = SimpleImputer(strategy=categorical_strategy)
            df[categorical_cols] = pd.DataFrame(imputer.fit_transform(df[categorical_cols]),
                                                columns=categorical_cols, index=df.index)
    return df

import pandas as pd
from typing import Optional, Dict, Tuple, List, Union
from sklearn.preprocessing import OneHotEncoder

def encode_data(
    df: pd.DataFrame,
    interval_cols: Optional[Dict[str, Tuple[List[float], List[str]]]] = None,
    categorical_mapping: Optional[Dict[str, Dict[str, Union[int, float]]]] = None,
    encoding_strategy: str = 'onehot'
) -> pd.DataFrame:
    """
    Encode les variables catégorielles et numériques dans un DataFrame.

    Args:
        df: DataFrame pandas à encoder
        interval_cols: Colonnes numériques à discrétiser, format:
                       {'col_name': ([bornes], [labels])}
        categorical_mapping: Mapping manuel des colonnes catégorielles
        encoding_strategy: Méthode d'encodage si pas de mapping fourni :
                           'onehot', 'label', 'ordinal'

    Returns:
        DataFrame encodé
    """
    df = df.copy()
    if interval_cols is None:
        interval_cols = {}
    if categorical_mapping is None:
        categorical_mapping = {}

    # 1. Encodage personnalisé
    for col, mapping in categorical_mapping.items():
        if col in df.columns:
            unknown_cats = set(df[col].dropna().unique()) - set(mapping.keys())
            if unknown_cats:
                raise ValueError(f"Catégories non mappées dans {col}: {unknown_cats}")
            df[col] = df[col].map(mapping)

    # 2. Encodage automatique des colonnes non mappées
    remaining_cat_cols = df.select_dtypes(include=['object', 'category']).columns
    auto_encode_cols = [col for col in remaining_cat_cols if col not in categorical_mapping]

    if encoding_strategy == 'onehot':
        if auto_encode_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[auto_encode_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(auto_encode_cols),
                index=df.index
            )
            df = df.drop(columns=auto_encode_cols)
            df = pd.concat([df, encoded_df], axis=1)

    elif encoding_strategy in ('label', 'ordinal'):
        for col in auto_encode_cols:
            df[col] = df[col].astype('category').cat.codes

    elif encoding_strategy != 'none':
        raise ValueError(f"Méthode d'encodage inconnue: {encoding_strategy}")

    # 3. Discrétisation (binning) des colonnes numériques
    for col, (bins, labels) in interval_cols.items():
        if col in df.columns:
            if len(bins) != len(labels) + 1:
                raise ValueError(f"Nombre de labels incorrect pour {col}. Attendu: {len(bins)-1}, Reçu: {len(labels)}")
            df[col] = pd.cut(df[col], bins=bins, labels=labels, right=False, include_lowest=True)

    return df

def normalize_data(df, columns=None, method='standard'):
    """
    Normalise les données numériques
    
    Args:
        df: DataFrame pandas
        columns: Liste des colonnes à normaliser (si None, normalise toutes les numériques)
        method: 'standard' (Z-score), 'minmax' (0-1), 'robust' (résistant aux outliers)
    
    Returns:
        DataFrame transformé
    """
    df = df.copy()
    
    # Sélection des colonnes numériques
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if not columns:
        return df
    
    # Sélection du scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Méthode inconnue: {method}. Choisir parmi 'standard', 'minmax', 'robust'")
    
    # Normalisation
    df[columns] = scaler.fit_transform(df[columns])
    
    return df


from typing import List, Optional, Union
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile,
    f_classif, f_regression,
    mutual_info_classif, mutual_info_regression
)

def select_features(
    df: pd.DataFrame,
    target_col: str,
    method: str = 'kbest',
    k: int = 5,
    percentile: int = 20,
    custom_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Sélectionne les meilleures features selon la méthode spécifiée.
    
    Args:
        df: DataFrame d'entrée
        target_col: Nom de la variable cible
        method: 'kbest', 'percentile', 'mutual_info', ou 'custom'
        k: Nombre de features à sélectionner (pour kbest)
        percentile: Pourcentage de features à garder (pour percentile)
        custom_features: Liste de features à garder (pour méthode 'custom')
    
    Returns:
        DataFrame avec les variables sélectionnées + la colonne cible
    """
    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' n'existe pas.")

    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if method == 'custom':
        if not custom_features:
            raise ValueError("Veuillez spécifier les features pour la méthode personnalisée.")
        missing = [col for col in custom_features if col not in X.columns]
        if missing:
            raise ValueError(f"Les colonnes suivantes sont absentes : {missing}")
        return df[custom_features + [target_col]]

    # Détermination du type de problème
    if y.dtype == 'object' or str(y.dtype).startswith("category"):
        y = y.astype('category').cat.codes
        score_func = f_classif
        mutual_func = mutual_info_classif
    else:
        score_func = f_regression
        mutual_func = mutual_info_regression

    if method == 'kbest':
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    elif method == 'percentile':
        selector = SelectPercentile(score_func=score_func, percentile=percentile)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_func, k=min(k, X.shape[1]))
    else:
        raise ValueError(f"Méthode inconnue: {method}. Choisir parmi 'kbest', 'percentile', 'mutual_info', 'custom'.")

    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return df[selected_features.tolist() + [target_col]]






