# test_pipeline.py

import pandas as pd
from app.core.preprocessing import handle_missing_values, encode_data, normalize_data, select_features
from app.core.visualisation import (
    plot_correlation_matrix, plot_histogram, plot_boxplot, 
    plot_categorical_count, plot_linear_relationship, 
    plot_missing_values, plot_unique_values
)

# 1. Charger un dataset d'exemple (ex: Iris dataset)
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.frame

# Ajouter une colonne catégorielle artificielle pour le test
df['type_fleur'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 2. Pipeline de prétraitement
print("🔵 Données originales:")
print(df.head())

# a. Gérer les valeurs manquantes
df = handle_missing_values(df)

# b. Encodage (exemple)
df = encode_data(df, categorical_mapping={'type_fleur': {'setosa': 0, 'versicolor': 1, 'virginica': 2}})

# c. Normalisation
df = normalize_data(df)

# d. Sélection de features
df = select_features(df, target_col='target', method='kbest', k=3)

print("✅ Données après prétraitement:")
print(df.head())

# 3. Visualisations
print("📊 Génération des graphiques...")

# Matrice de corrélation
plot_correlation_matrix(df)

# Histogramme sur une colonne numérique
plot_histogram(df, column=df.columns[0])

# Boxplot sur une colonne numérique
plot_boxplot(df, column=df.columns[1])

# (Pas de variables catégorielles après encodage et sélection, mais tu pourrais ajouter un test ici)

# Nuage de points entre deux colonnes
plot_linear_relationship(df, x_col=df.columns[0], y_col=df.columns[1])

# Valeurs manquantes
plot_missing_values(df)

# Valeurs uniques
plot_unique_values(df)

print("🎉 Test global terminé. Graphiques sauvegardés dans le dossier courant.")
