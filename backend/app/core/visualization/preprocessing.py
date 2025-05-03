from .charts import chart
import pandas as pd
from typing import Optional

def plot_correlation_matrix(df: pd.DataFrame) -> str:
    """Matrice de corrélation avec heatmap"""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError("Aucune colonne numérique pour la matrice de corrélation")
    return chart.create_heatmap(
        matrix=numeric_df.corr(),
        title="Matrice de Corrélation"
    )

def plot_histogram(df: pd.DataFrame, column: str) -> str:
    """Histogramme pour une colonne numérique"""
    if column not in df.columns:
        raise ValueError(f"Colonne {column} non trouvée")
    return chart.create_histogram(
        data=df[column],
        title=f"Distribution de {column}",
        xlabel=column
    )

def plot_boxplot(df: pd.DataFrame, column: str) -> str:
    """Boxplot pour une colonne numérique"""
    if column not in df.columns:
        raise ValueError(f"Colonne {column} non trouvée")
    return chart.create_boxplot(
        data=df[column],
        title=f"Boxplot de {column}",
        xlabel=column
    )

def plot_categorical_count(df: pd.DataFrame, column: str) -> str:
    """Countplot pour une colonne catégorielle"""
    if column not in df.columns:
        raise ValueError(f"Colonne {column} non trouvée")
    return chart.create_countplot(
        df=df,
        column=column,
        title=f"Distribution de {column}"
    )

def plot_linear_relationship(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    """Nuage de points avec régression linéaire"""
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Colonnes spécifiées non trouvées")
    return chart.create_scatterplot(
        df=df,
        x_col=x_col,
        y_col=y_col,
        title=f"Relation entre {x_col} et {y_col}"
    )

def plot_missing_values(df: pd.DataFrame) -> str:
    """Visualisation des valeurs manquantes"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        raise ValueError("Aucune valeur manquante détectée")
    return chart.create_barplot(
        data=missing,
        title="Valeurs manquantes par colonne",
        xlabel="Nombre de valeurs manquantes",
        ylabel="Colonnes"
    )

def plot_unique_values(df: pd.DataFrame) -> str:
    """Visualisation des valeurs uniques"""
    unique_counts = df.nunique().sort_values(ascending=False)
    return chart.create_barplot(
        data=unique_counts,
        title="Valeurs uniques par colonne",
        xlabel="Nombre de valeurs uniques",
        ylabel="Colonnes"
    )