import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def handle_missing_values(df, numeric_strategy='mean', categorical_strategy='constant'):
    """
    Gère les valeurs manquantes dans un DataFrame.
    
    Args:
        df: DataFrame pandas
        numeric_strategy: 'mean', 'median' ou 'most_frequent'
        categorical_strategy: 'constant' ou 'most_frequent'
    
    Returns:
        DataFrame transformé
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
from typing import Dict, List, Union, Optional

def encode_data(df: pd.DataFrame, 
                interval_cols: Optional[Dict[str, tuple]] = None, 
                categorical_mapping: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    """
    Encode les variables catégorielles et numériques dans un DataFrame.
    
    Args:
        df: DataFrame pandas à encoder
        interval_cols: Dictionnaire des colonnes numériques à discrétiser.
                      Format: {'col_name': ([bornes], [labels])}
                      Ex: {'age': ([0, 18, 65, 100], ['enfant', 'adulte', 'senior'])}
        categorical_mapping: Dictionnaire des mappings pour les colonnes catégorielles.
                            Format: {'col_name': {'cat1': val1, 'cat2': val2}}
                            Ex: {'gender': {'M': 0, 'F': 1}}
    
    Returns:
        DataFrame transformé avec les encodages appliqués
        
    Exemple:
        >>> df = pd.DataFrame({'age': [10, 25, 70], 'gender': ['M', 'F', 'M']})
        >>> encoded = encode_data(
        ...     df,
        ...     interval_cols={'age': ([0, 18, 65, 100], ['enfant', 'adulte', 'senior'])},
        ...     categorical_mapping={'gender': {'M': 0, 'F': 1}}
        ... )
    """
    df = df.copy()

    # Validation des inputs
    if interval_cols is None:
        interval_cols = {}
    if categorical_mapping is None:
        categorical_mapping = {}

    # Encodage des variables catégorielles
    for col, mapping in categorical_mapping.items():
        if col in df.columns:
            # Vérification des catégories manquantes
            unknown_cats = set(df[col].unique()) - set(mapping.keys())
            if unknown_cats:
                raise ValueError(f"Catégories non mappées dans {col}: {unknown_cats}")
            
            df[col] = df[col].map(mapping)
            # Conversion en type catégoriel si valeurs sont discrètes
            if all(isinstance(v, (int, float)) for v in mapping.values()):
               pd.to_numeric(df[col], errors='ignore')
    # Découpage en intervalles pour les variables numériques
    for col, (bins, labels) in interval_cols.items():
        if col in df.columns:
            if len(bins) != len(labels) + 1:
                raise ValueError(f"Nombre de labels incorrect pour {col}. Attendu: {len(bins)-1}, Reçu: {len(labels)}")
            
            try:
                df[col] = pd.cut(df[col], 
                                bins=bins, 
                                labels=labels, 
                                right=False, 
                                include_lowest=True)
            except Exception as e:
                raise ValueError(f"Erreur lors du découpage de {col}: {str(e)}")

    return df





from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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



from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

def select_features(df, target_col=None, method='kbest', k=5, estimator=None):
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
    from sklearn.ensemble import RandomForestClassifier

    if target_col and target_col not in df.columns:
        raise ValueError(f"La colonne cible {target_col} n'existe pas")
    
    if method not in ['kbest', 'mutual_info', 'rfe']:
        raise ValueError("Méthode non supportée. Choisir: 'kbest', 'mutual_info', 'rfe'")
    
    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df.copy()
        y = None
    
    if method == 'kbest':
        selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
    elif method == 'rfe':
        estimator = estimator or RandomForestClassifier()
        selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
    
    if y is not None:
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()]
        selected_df = X[selected_cols].copy()
        selected_df[target_col] = y.values  # pour éviter le warning SettingWithCopy
    else:
        selected_cols = X.columns[:k]
        selected_df = X[selected_cols].copy()
    
    return selected_df





