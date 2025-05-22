import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import uuid

# Dossier où enregistrer les graphes
OUTPUT_DIR = "app/data/resultats"

# S'assurer que le dossier de résultats existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(fig, name_prefix):
    filename = f"{name_prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots()
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    return filepath

# Visualisation de la matrice de corrélation
def plot_correlation_matrix(df):
    # Sélectionner uniquement les colonnes numériques
    df_numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df_numeric.corr()  # Calcul de la corrélation uniquement sur les colonnes numériques
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Heatmap des corrélations")
    return save_plot(fig, "correlation_matrix")

# Visualisation de l'histogramme pour une colonne
def plot_histogram(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), kde=True, ax=ax)  # Utilisation de dropna() pour éviter les valeurs manquantes
    ax.set_title(f"Histogramme de {column}")
    return save_plot(fig, "histogram")

# Visualisation du boxplot pour une colonne
def plot_boxplot(df, column):
    fig, ax = plt.subplots()
    sns.boxplot(y=df[column].dropna(), ax=ax)  # Utilisation de dropna() pour éviter les valeurs manquantes
    ax.set_title(f"Boxplot de {column}")
    return save_plot(fig, "boxplot")

# Visualisation de la distribution des valeurs manquantes
def plot_missing_values(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        raise ValueError("Aucune valeur manquante à afficher.")
    sns.barplot(x=missing.index, y=missing.values, ax=ax)
    ax.set_ylabel("Pourcentage de valeurs manquantes")
    plt.xticks(rotation=45)
    ax.set_title("Valeurs manquantes par colonne")
    return save_plot(fig, "missing_values")

# Visualisation des valeurs uniques par colonne
def plot_unique_values(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_counts = df.nunique().sort_values(ascending=False)
    sns.barplot(x=unique_counts.index, y=unique_counts.values, ax=ax)
    ax.set_ylabel("Nombre de valeurs uniques")
    plt.xticks(rotation=45)
    ax.set_title("Valeurs uniques par colonne")
    return save_plot(fig, "unique_values")


# Ajoutez cette fonction à la fin de preprocessing.py

def plot_categorical_count(df, column):
    """Visualise le décompte des catégories pour une colonne donnée."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=column, data=df, ax=ax)
    ax.set_title(f"Distribution de {column}")
    plt.xticks(rotation=45)
    return save_plot(fig, "categorical_count")

def plot_linear_relationship(df, x_col, y_col):
    """Visualise la relation linéaire entre deux variables."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=x_col, y=y_col, data=df, ax=ax, line_kws={"color": "red"})
    ax.set_title(f"Relation linéaire entre {x_col} et {y_col}")
    plt.xticks(rotation=45)
    return save_plot(fig, "linear_relationship")