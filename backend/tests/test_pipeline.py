import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from app.api.upload_routes import upload_file
from app.core.preprocessing import handle_missing_values, encode_categorical_variables, select_features
from app.results.visualisation import (
    plot_correlation_matrix,
    plot_histogram,
    plot_boxplot,
    plot_categorical_count,
    plot_linear_relationship,
    plot_missing_values,
    plot_unique_values
)

def test_pipeline():
    # Étape 1 : Lecture manuelle du fichier JSON (iris dataset)
    file_path = os.path.join(os.path.dirname(__file__), "iris.json")
    df = pd.read_json(file_path)

    # Étape 2 : Simuler un upload avec un DummyFile
    class DummyFile:
        def __init__(self, real_path, filename):
            self.real_path = real_path  # vrai chemin vers iris.json
            self.filename = filename    # nom fictif pour l'upload

        def save(self, path):
            with open(self.real_path, 'rb') as src, open(path, 'wb') as dst:
                dst.write(src.read())

    dummy_file = DummyFile(file_path, "iris.json")
    csv_path, error = upload_file(dummy_file)

    if error:
        print("❌ Erreur upload :", error)
        return

    print(f"📄 CSV généré : {csv_path}")
    print("✅ Fichier existe ?", os.path.exists(csv_path))

    # Étape 3 : Chargement du CSV uploadé (converti depuis JSON)
    df = pd.read_csv(csv_path)

    # Étape 4 : Nettoyage des valeurs manquantes
    df = handle_missing_values(df)

    # Étape 5 : Sélection de variables (target = 'species')
    if 'species' in df.columns:
        df = select_features(df, target_col='species', method='kbest', k=3)
    else:
        print("⚠️ Colonne 'species' introuvable, saut de la sélection de variables")

    # Étape 6 : Encodage des variables catégorielles
    df = encode_categorical_variables(df)

    # Étape 7 : Visualisations
    plot_missing_values(df)
    plot_unique_values(df)
    plot_correlation_matrix(df)

    for col in df.select_dtypes(include='number').columns:
        plot_histogram(df, col)
        plot_boxplot(df, col)

    for col in df.select_dtypes(include='object').columns:
        plot_categorical_count(df, col)

    if 'species' in df.columns:
        for col in df.columns:
            if col != 'species' and df[col].dtype in ["float64", "int64"]:
                plot_linear_relationship(df, col, 'species')

    # Étape 8 : Exporter la donnée prétraitée
    output_path = os.path.join("backend", "app", "data", "processed", "data_pretraitee.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Données prétraitées sauvegardées dans : {output_path}")

if __name__ == '__main__':
    test_pipeline()
    print("✅ Pipeline de test terminé avec succès.")
