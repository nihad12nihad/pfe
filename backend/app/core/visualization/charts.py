#core/visualization/charts.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import uuid
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

# Constantes
OUTPUT_DIR = "app/data/resultats"  # Répertoire de sortie

# Fonction pour sauvegarder les graphiques
def save_plot(fig, name_prefix):
    filename = f"{name_prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots()  
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    return filepath

# --- Visualisation des résultats des modèles ---
def plot_confusion_matrix(df, y_true, y_pred):
    """ Trace une matrice de confusion pour les modèles de classification. """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df.columns, yticklabels=df.columns)
    plt.title('Matrice de confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Réel')
    image_path = "app/data/resultats/images/confusion_matrix.png"
    plt.savefig(image_path)
    plt.close()
    return image_path

def plot_classification_results(y_true, y_pred, y_score):
    """ Trace les résultats de classification : courbe ROC et Bar plot des scores. """
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion')
    image_path_cm = "app/data/resultats/images/confusion_matrix.png"
    plt.savefig(image_path_cm)
    plt.close()

    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Courbe ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Courbe ROC')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.legend(loc='lower right')
    image_path_roc = "app/data/resultats/images/roc_curve.png"
    plt.savefig(image_path_roc)
    plt.close()

    return image_path_cm, image_path_roc

def plot_regression_results(y_true, y_pred):
    """ Trace les résultats de régression : courbe des résidus et vraie vs prédite. """
    # Résidus
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Résidus de la régression')
    plt.xlabel('Prédictions')
    plt.ylabel('Résidus')
    image_path_residuals = "sapp/data/resultats/images/residuals.png"
    plt.savefig(image_path_residuals)
    plt.close()

    # Vraies vs Prédites
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.title('Vraies valeurs vs Prédites')
    plt.xlabel('Vraies valeurs')
    plt.ylabel('Prédites')
    image_path_true_vs_pred = "sapp/data/resultats/images/true_vs_pred.png"
    plt.savefig(image_path_true_vs_pred)
    plt.close()

    return image_path_residuals, image_path_true_vs_pred

def plot_clustering_results(df, n_clusters=3):
    """ Trace les résultats du clustering (ici avec KMeans). """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue='Cluster', palette='Set1', data=df)
    plt.title('Visualisation des clusters')
    image_path_clusters = "app/data/resultats/images/clusters.png"
    plt.savefig(image_path_clusters)
    plt.close()

    return image_path_clusters

def plot_association_rules(df):
    """ Trace un graphique des règles d'association (ex. avec réseau de règles). """
    # Cette fonction nécessiterait un peu plus de logique pour exploiter un algorithme comme Apriori
    # Pour cet exemple, je vais simplement tracer un réseau fictif.
    plt.figure(figsize=(8, 6))
    # Vous pouvez adapter ici pour intégrer des visualisations plus complexes de règles
    plt.title('Règles d\'association (exemple)')
    image_path_association = "app/data/resultats/images/association_rules.png"
    plt.savefig(image_path_association)
    plt.close()

    return image_path_association
