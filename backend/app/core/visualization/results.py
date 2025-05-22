import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_samples, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve
from sklearn.tree import plot_tree
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import os

# ---------------------- CLASSIFICATION ----------------------

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title("Matrice de confusion")
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_true, y_scores, pos_label=1):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt.gcf()

def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Précision-Rappel")
    plt.tight_layout()
    return plt.gcf()

def plot_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(feature_names)[indices], importances[indices], color="skyblue")
        plt.xlabel("Importance")
        plt.title("Importance des variables")
        plt.tight_layout()
        return plt.gcf()
    else:
        raise ValueError("Le modèle ne fournit pas d'attribut 'feature_importances_'")

def plot_2d_projection(X, y, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "Projection PCA"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
        title = "Projection t-SNE"
    else:
        raise ValueError("Méthode non supportée : choisir 'pca' ou 'tsne'.")

    X_reduced = reducer.fit_transform(X)
    df_proj = pd.DataFrame({
        'X1': X_reduced[:, 0],
        'X2': X_reduced[:, 1],
        'Label': y
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_proj, x='X1', y='X2', hue='Label', palette='Set2', alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_learning_curve(estimator, X, y, scoring="accuracy", cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Train")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation")
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Taille de l'échantillon d'entraînement")
    plt.ylabel(scoring.capitalize())
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def plot_decision_tree(model, feature_names, class_names):
    if hasattr(model, "tree_"):
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=feature_names, class_names=class_names,
                  filled=True, rounded=True, fontsize=10)
        plt.title("Arbre de Décision")
        plt.tight_layout()
        return plt.gcf()
    else:
        raise ValueError("Le modèle n’est pas un arbre de décision.")

def plot_feature_correlation(X, title="Matrice de corrélation"):
    plt.figure(figsize=(10, 8))
    corr = pd.DataFrame(X).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

# Ajoutez ici les fonctions de sauvegarde si vous souhaitez enregistrer automatiquement les figures.

# ---------------------- REGRESSION ----------------------

def plot_regression_results(y_true, y_pred, save_path=None):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6, color='teal')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title("Réel vs Prédit (Régression)")
    plt.tight_layout()
    fig = plt.gcf()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    return fig


def plot_regression_errors(y_true, y_pred, save_path=None):
    errors = np.array(y_true) - np.array(y_pred)
    plt.figure()
    sns.histplot(errors, kde=True, color='orange')
    plt.title("Distribution des erreurs de prédiction")
    plt.xlabel("Erreur")
    plt.tight_layout()
    fig = plt.gcf()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    return fig


def plot_regression_coefficients(coefficients: dict, intercept: float = None, title: str = "Coefficients de régression", save_path=None):
    features = list(coefficients.keys())
    values = list(coefficients.values())

    plt.figure(figsize=(8, 4))
    bars = plt.bar(features, values, color='skyblue')
    plt.axhline(0, color='grey', linewidth=0.8)
    plt.title(title + (f" (Intercept = {intercept:.2f})" if intercept is not None else ""))
    plt.xlabel("Variables explicatives")
    plt.ylabel("Coefficients")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

    fig = plt.gcf()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    return fig


def plot_residuals(y_true, y_pred, save_path=None):
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.6, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Résidus (erreurs)")
    plt.title("Résidus vs Valeurs prédites")
    plt.grid(True)
    plt.tight_layout()
    fig = plt.gcf()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    return fig


def plot_3d_regression(X_test, y_true, y_pred, features=None, save_path=None):
    if X_test.shape[1] < 2:
        raise ValueError("La visualisation 3D nécessite au moins 2 variables explicatives")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = X_test.iloc[:, 0]
    ys = X_test.iloc[:, 1]

    ax.scatter(xs, ys, y_true, color='blue', label='Réel', alpha=0.6)
    ax.scatter(xs, ys, y_pred, color='red', label='Prédit', alpha=0.6)

    ax.set_xlabel(features[0] if features else "Feature 1")
    ax.set_ylabel(features[1] if features else "Feature 2")
    ax.set_zlabel("Cible")
    ax.set_title("Visualisation 3D Réel vs Prédit")

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)

    return fig
# ---------------------- CLUSTERING ----------------------

def plot_clusters_2d(X, labels, method='pca'):
    reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, random_state=42)
    reduced = reducer.fit_transform(X)
    plt.figure()
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='Set2')
    plt.title(f"Clustering visualisé en 2D ({method.upper()})")
    plt.tight_layout()
    return plt.gcf()

def plot_silhouette_scores(X, labels):
    silhouette_vals = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)

    plt.figure()
    sns.histplot(silhouette_vals, bins=20, kde=True)
    plt.axvline(silhouette_avg, color="red", linestyle="--", label=f"Score moyen = {silhouette_avg:.2f}")
    plt.title("Distribution des scores de silhouette")
    plt.xlabel("Score de silhouette")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def plot_cluster_centers(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, method: str = "pca") -> plt.Figure:
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    X_2d = reducer.fit_transform(X)
    centers_2d = reducer.transform(centers)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette='Set2', legend="full")
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], s=200, c='black', marker='X', label='Centres')
    plt.title(f"Clusters + Centres ({method.upper()})")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def plot_cluster_distribution(labels: np.ndarray) -> plt.Figure:
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(cluster_sizes.keys()), y=list(cluster_sizes.values()), palette="viridis")
    plt.xlabel("Cluster")
    plt.ylabel("Nombre de points")
    plt.title("Taille de chaque cluster")
    plt.tight_layout()
    return plt.gcf()

def plot_dendrogram(X: np.ndarray, method: str = 'ward') -> plt.Figure:
    distance_matrix = pdist(X)
    linkage_matrix = linkage(distance_matrix, method=method)

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title(f"Dendrogramme (méthode: {method})")
    plt.xlabel("Index des échantillons")
    plt.ylabel("Distance")
    plt.tight_layout()
    return plt.gcf()

# ---------------------- ASSOCIATION ----------------------

def plot_association_rules_heatmap(rules_df, metric='lift', itemsets_df=None, top_n_graph=20, top_n_itemsets=10):
    """
    Génère plusieurs visualisations pour les règles d'association :
    - Heatmap (metric entre antécédents et conséquents)
    - Graphe des règles (réseau)
    - Barplot des itemsets fréquents
    - Scatter plot support vs confidence vs lift

    Args:
        rules_df (pd.DataFrame): Règles d'association.
        metric (str): Métrique à utiliser (lift, confidence...).
        itemsets_df (pd.DataFrame): Optionnel — DataFrame des itemsets fréquents.
        top_n_graph (int): Nombre de règles à afficher dans le graphe.
        top_n_itemsets (int): Nombre d’itemsets dans le barplot.

    Returns:
        dict[str, matplotlib.figure.Figure]: Dictionnaire de figures.
    """
    figures = {}

    # ----- 1. Heatmap -----
    try:
        pivot_table = rules_df.pivot_table(
            index='antecedents', columns='consequents', values=metric, aggfunc='mean', fill_value=0
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Heatmap des règles d'association ({metric})")
        plt.ylabel("Antécédents")
        plt.xlabel("Conséquents")
        plt.tight_layout()
        figures["heatmap"] = plt.gcf()
    except Exception as e:
        print(f"[Erreur Heatmap] {e}")

    # ----- 2. Graphe des règles -----
    try:
        graph_df = rules_df.sort_values(by=metric, ascending=False).head(top_n_graph)
        G = nx.DiGraph()
        for _, row in graph_df.iterrows():
            antecedent = ', '.join(map(str, row['antecedents']))
            consequent = ', '.join(map(str, row['consequents']))
            G.add_edge(antecedent, consequent, weight=row[metric])

        pos = nx.spring_layout(G, seed=42)
        weights = [G[u][v]['weight'] for u, v in G.edges()]

        plt.figure(figsize=(12, 8))
        nx.draw(
            G, pos, with_labels=True, node_color='skyblue', edge_color=weights,
            width=2.0, edge_cmap=plt.cm.coolwarm, node_size=2000, font_size=10
        )
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(min(weights), max(weights)))
        plt.colorbar(sm, label=metric)
        plt.title(f"Graphe des règles d'association ({metric})")
        plt.tight_layout()
        figures["graph"] = plt.gcf()
    except Exception as e:
        print(f"[Erreur Graphe] {e}")

    # ----- 3. Barplot des itemsets fréquents -----
    if itemsets_df is not None:
        try:
            df = itemsets_df.copy()
            df['itemsets'] = df['itemsets'].apply(lambda x: ', '.join(map(str, x)))
            df = df.sort_values(by='support', ascending=False).head(top_n_itemsets)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='support', y='itemsets', palette='viridis')
            plt.title("Top itemsets fréquents (support)")
            plt.xlabel("Support")
            plt.ylabel("Itemsets")
            plt.tight_layout()
            figures["barplot_itemsets"] = plt.gcf()
        except Exception as e:
            print(f"[Erreur Barplot Itemsets] {e}")

    # ----- 4. Scatterplot support vs confidence vs lift -----
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=rules_df,
            x='support', y='confidence', size='lift', hue='lift',
            palette='viridis', sizes=(20, 200), legend="brief"
        )
        plt.title("Règles d'association : support vs confiance (taille/couleur = lift)")
        plt.xlabel("Support")
        plt.ylabel("Confiance")
        plt.legend()
        plt.tight_layout()
        figures["scatter"] = plt.gcf()
    except Exception as e:
        print(f"[Erreur Scatter Plot] {e}")

    return figures