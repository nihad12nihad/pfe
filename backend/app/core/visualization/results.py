from .charts import chart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

def plot_classification_results(metrics: Dict[str, float], algorithm_name: str) -> str:
    """Visualisation des métriques de classification"""
    if not metrics:
        raise ValueError("Aucune métrique fournie")
    
    fig, ax = chart._prepare_figure()
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax)
    ax.bar_label(ax.containers[0], fmt='%.2f')
    chart._apply_common_styling(
        ax,
        title=f"Performance de {algorithm_name}",
        xlabel="Métriques",
        ylabel="Score"
    )
    ax.set_ylim(0, 1)
    return chart._save_figure(fig, f"classification_{algorithm_name}", "results")

def plot_regression_results(metrics: Dict[str, float], algorithm_name: str) -> str:
    """Visualisation des métriques de régression"""
    if not metrics:
        raise ValueError("Aucune métrique fournie")
    
    df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    fig, ax = chart._prepare_figure()
    sns.barplot(x='Value', y='Metric', data=df, ax=ax)
    chart._apply_common_styling(
        ax,
        title=f"Performance de {algorithm_name} (Régression)",
        xlabel="Score",
        ylabel="Métriques"
    )
    return chart._save_figure(fig, f"regression_{algorithm_name}", "results")

def plot_confusion_matrix(cm: np.ndarray, labels: list) -> str:
    """Matrice de confusion visuelle"""
    if cm.size == 0:
        raise ValueError("Matrice de confusion vide")
    
    fig, ax = chart._prepare_figure()
    sns.heatmap(cm, annot=True, fmt="d", 
               xticklabels=labels, 
               yticklabels=labels, 
               cmap="Blues", cbar=False,
               ax=ax)
    chart._apply_common_styling(
        ax,
        title="Matrice de confusion",
        xlabel="Prédit",
        ylabel="Réel"
    )
    return chart._save_figure(fig, "confusion_matrix", "results")

def plot_clusters(data: np.ndarray, labels: np.ndarray, title: str = "Clusters") -> str:
    """Visualisation 2D des clusters"""
    if data.shape[1] < 2:
        raise ValueError("Les données doivent avoir au moins 2 dimensions")
    
    fig, ax = chart._prepare_figure()
    scatter = ax.scatter(data[:,0], data[:,1], c=labels, 
                        cmap="viridis", alpha=0.6, 
                        edgecolors='w', linewidths=0.5)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    chart._apply_common_styling(
        ax,
        title=title,
        xlabel="Dimension 1",
        ylabel="Dimension 2"
    )
    return chart._save_figure(fig, "clusters", "clustering")
