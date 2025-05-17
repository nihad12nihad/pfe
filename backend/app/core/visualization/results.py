import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import numpy as np
from IPython.display import display


# Classification

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_classification_scores(scores_dict):
    names = list(scores_dict.keys())
    values = list(scores_dict.values())
    plt.figure(figsize=(8, 6))
    sns.barplot(x=names, y=values)
    plt.title("Scores de classification")
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return  # Pas applicable
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title("Importance des variables")
    plt.tight_layout()
    plt.show()

# Régression

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True)
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Résidus")
    plt.title("Courbe des résidus")
    plt.tight_layout()
    plt.show()

def plot_true_vs_pred(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title("Vrai vs Prédit")
    plt.tight_layout()
    plt.show()

def plot_regression_errors(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    scores = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}
    plot_classification_scores(scores)

# Clustering

def plot_clusters_2d(X, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title("Visualisation des clusters (2D)")
    plt.tight_layout()
    plt.show()

def plot_silhouette(X, labels):
    silhouette_vals = silhouette_samples(X, labels)
    plt.figure(figsize=(8, 6))
    sns.histplot(silhouette_vals, bins=20, kde=True)
    plt.title("Silhouette plot")
    plt.xlabel("Score de silhouette")
    plt.tight_layout()
    plt.show()

def plot_dendrogram(X):
    linked = linkage(X, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title("Dendrogramme (clustering hiérarchique)")
    plt.tight_layout()
    plt.show()

# Association

def plot_association_rules_graph(rules):
    import networkx as nx
    G = nx.DiGraph()
    for i, rule in rules.iterrows():
        for antecedent in rule['antecedents']:
            for consequent in rule['consequents']:
                G.add_edge(antecedent, consequent, weight=rule['lift'])
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue",
            font_size=10, edge_color=edge_weights, edge_cmap=plt.cm.Blues,
            width=2, arrows=True)
    plt.title("Graphes des règles d'association")
    plt.tight_layout()
    plt.show()

def display_rules_table(rules):
    display(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
