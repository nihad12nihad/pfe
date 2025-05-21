# app/core/visualization/charts.py
import os
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Répertoire de sortie
OUTPUT_DIR = "app/data/resultats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_correlation_matrix(df):
    df_numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Heatmap des corrélations")
    return fig

def plot_histogram(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(f"Histogramme de {column}")
    return fig

def plot_boxplot(df, column):
    fig, ax = plt.subplots()
    sns.boxplot(y=df[column].dropna(), ax=ax)
    ax.set_title(f"Boxplot de {column}")
    return fig

def plot_missing_values(df):
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=missing.index, y=missing.values, ax=ax)
    ax.set_ylabel("Pourcentage de valeurs manquantes")
    ax.set_title("Valeurs manquantes par colonne")
    plt.xticks(rotation=45)
    return fig

def plot_unique_values(df):
    unique_counts = df.nunique().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=unique_counts.index, y=unique_counts.values, ax=ax)
    ax.set_ylabel("Nombre de valeurs uniques")
    ax.set_title("Valeurs uniques par colonne")
    plt.xticks(rotation=45)
    return fig
