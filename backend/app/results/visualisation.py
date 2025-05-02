import matplotlib
matplotlib.use('Agg')  # Utilise un mode sans fenêtre graphique

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Matrice de corrélation (Heatmap)
def plot_correlation_matrix(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Matrice de Corrélation')
    plt.savefig('backend/static/graphique1.png')

# 2. Histogramme de distribution
def plot_histogram(df: pd.DataFrame, column: str):
    plt.figure(figsize=(8,6))
    sns.histplot(df[column].dropna(), bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution de {column}')
    plt.xlabel(column)
    plt.ylabel('Fréquence')
    plt.savefig('backend/static/graphique2.png')

# 3. Boxplot pour chaque variable numérique
def plot_boxplot(df: pd.DataFrame, column: str):
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[column], color='lightcoral')
    plt.title(f'Boxplot de {column}')
    plt.xlabel(column)
    plt.savefig('backend/static/graphique3.png')

# 4. Countplot pour variables catégorielles
def plot_categorical_count(df: pd.DataFrame, column: str):
    plt.figure(figsize=(8,6))
    sns.countplot(x=column, data=df, palette='muted')
    plt.title(f'Distribution de {column}')
    plt.xlabel(column)
    plt.ylabel('Nombre')
    plt.xticks(rotation=45)
    plt.savefig('backend/static/graphique4.png')

# 5. Nuage de points avec droite de régression
def plot_linear_relationship(df: pd.DataFrame, x_col: str, y_col: str):
    plt.figure(figsize=(8,6))
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.title(f'Relation linéaire entre {x_col} et {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.savefig('backend/static/graphique5.png')

# 6. Histogramme des valeurs manquantes
def plot_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("✅ Pas de valeurs manquantes détectées.")
        return
    plt.figure(figsize=(10,6))
    sns.barplot(x=missing.values, y=missing.index, palette='viridis')
    plt.title('Valeurs manquantes par colonne')
    plt.xlabel('Nombre de valeurs manquantes')
    plt.ylabel('Colonnes')
    plt.savefig('backend/static/graphique6.png')

# 7. Nombre de valeurs uniques par colonne
def plot_unique_values(df: pd.DataFrame):
    unique_counts = df.nunique().sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=unique_counts.values, y=unique_counts.index, palette='crest')
    plt.title('Nombre de valeurs uniques par colonne')
    plt.xlabel('Nombre de valeurs uniques')
    plt.ylabel('Colonnes')
    plt.savefig('backend/static/graphique7.png')
