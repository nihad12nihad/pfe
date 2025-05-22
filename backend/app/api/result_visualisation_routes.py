from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

<<<<<<< HEAD
from app.core.visualization.charts import (
    plot_confusion_matrix,
    plot_classification_results,
    plot_regression_results,
    plot_clustering_results,
    plot_association_rules
)
=======
import app.core.visualization.results as vis  # ton module avec toutes les fonctions
>>>>>>> f440206c5d1d41dcbf0adf5979fddaaaa5776bbb

router = APIRouter()

OUTPUT_DIR = "app/data/resultats"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, prefix: str) -> str:
    """Sauvegarde la figure dans OUTPUT_DIR avec un nom unique et retourne le chemin."""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


class ClassificationData(BaseModel):
    y_true: List[int]
    y_pred: List[int]
    y_scores: Optional[List[float]] = None
    labels: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    model_params: Optional[Dict[str, Any]] = None
    class_names: Optional[List[str]] = None
    X: Optional[List[List[float]]] = None  # pour projection 2D ou learning curve


class RegressionData(BaseModel):
    y_true: List[float]
    y_pred: List[float]
    coefficients: Optional[Dict[str, float]] = None
    intercept: Optional[float] = None
    X_test: Optional[List[List[float]]] = None
    feature_names: Optional[List[str]] = None


class ClusteringData(BaseModel):
    X: List[List[float]]
    labels: List[int]
    centers: Optional[List[List[float]]] = None
    method: Optional[str] = 'pca'


# --- Routes Classification ---

@router.post("/classification/confusion_matrix")
async def classification_confusion_matrix(data: ClassificationData):
    fig = vis.plot_confusion_matrix(data.y_true, data.y_pred, labels=data.labels)
    path = save_figure(fig, "confusion_matrix")
    return {"image_path": path}


@router.post("/classification/roc_curve")
async def classification_roc_curve(data: ClassificationData):
    if not data.y_scores:
        raise HTTPException(status_code=400, detail="y_scores requis pour courbe ROC")
    fig = vis.plot_roc_curve(data.y_true, data.y_scores)
    path = save_figure(fig, "roc_curve")
    return {"image_path": path}


@router.post("/classification/precision_recall")
async def classification_precision_recall(data: ClassificationData):
    if not data.y_scores:
        raise HTTPException(status_code=400, detail="y_scores requis pour courbe Précision-Rappel")
    fig = vis.plot_precision_recall_curve(data.y_true, data.y_scores)
    path = save_figure(fig, "precision_recall")
    return {"image_path": path}


@router.post("/classification/feature_importance")
async def classification_feature_importance(data: ClassificationData):
    if not data.model_params or 'feature_importances_' not in data.model_params:
        raise HTTPException(status_code=400, detail="feature_importances_ requis dans model_params")
    class DummyModel:
        feature_importances_ = data.model_params['feature_importances_']

    fig = vis.plot_feature_importance(DummyModel(), data.feature_names or [])
    path = save_figure(fig, "feature_importance")
    return {"image_path": path}


@router.post("/classification/2d_projection")
async def classification_2d_projection(data: ClassificationData, method: Optional[str] = 'pca'):
    if not data.X or not data.y_true:
        raise HTTPException(status_code=400, detail="X et y_true requis")
    X = np.array(data.X)
    y = np.array(data.y_true)
    fig = vis.plot_2d_projection(X, y, method=method)
    path = save_figure(fig, f"2d_projection_{method}")
    return {"image_path": path}


@router.post("/classification/learning_curve")
async def classification_learning_curve(data: ClassificationData):
    if not data.X or not data.y_true or not data.model_params or 'estimator' not in data.model_params:
        raise HTTPException(status_code=400, detail="X, y_true et estimator requis")
    from sklearn.base import BaseEstimator

    estimator = data.model_params['estimator']
    # Ici on suppose que l'estimateur est déjà entraîné passé sous forme sérialisée ? Sinon, adapter l'appel

    fig = vis.plot_learning_curve(estimator, np.array(data.X), np.array(data.y_true))
    path = save_figure(fig, "learning_curve")
    return {"image_path": path}


@router.post("/classification/decision_tree")
async def classification_decision_tree(data: ClassificationData):
    if not data.model_params or 'model' not in data.model_params:
        raise HTTPException(status_code=400, detail="model requis")
    model = data.model_params['model']
    fig = vis.plot_decision_tree(model, data.feature_names or [], data.class_names or [])
    path = save_figure(fig, "decision_tree")
    return {"image_path": path}


# --- Routes Regression ---

@router.post("/regression/results")
async def regression_results(data: RegressionData):
    fig = vis.plot_regression_results(data.y_true, data.y_pred)
    path = save_figure(fig, "regression_results")
    return {"image_path": path}


@router.post("/regression/errors")
async def regression_errors(data: RegressionData):
    fig = vis.plot_regression_errors(data.y_true, data.y_pred)
    path = save_figure(fig, "regression_errors")
    return {"image_path": path}


@router.post("/regression/coefficients")
async def regression_coefficients(data: RegressionData):
    if not data.coefficients:
        raise HTTPException(status_code=400, detail="coefficients requis")
    fig = vis.plot_regression_coefficients(data.coefficients, data.intercept)
    path = save_figure(fig, "regression_coefficients")
    return {"image_path": path}


@router.post("/regression/residuals")
async def regression_residuals(data: RegressionData):
    fig = vis.plot_residuals(data.y_true, data.y_pred)
    path = save_figure(fig, "regression_residuals")
    return {"image_path": path}


@router.post("/regression/3d_visualization")
async def regression_3d_visualization(data: RegressionData):
    if not data.X_test:
        raise HTTPException(status_code=400, detail="X_test requis")
    X_test = pd.DataFrame(data.X_test)
    fig = vis.plot_3d_regression(X_test, data.y_true, data.y_pred, data.feature_names)
    path = save_figure(fig, "regression_3d")
    return {"image_path": path}


# --- Routes Clustering ---

@router.post("/clustering/2d_clusters")
async def clustering_2d_clusters(data: ClusteringData):
    X = np.array(data.X)
    labels = np.array(data.labels)
    fig = vis.plot_clusters_2d(X, labels, method=data.method)
    path = save_figure(fig, "clustering_2d")
    return {"image_path": path}


@router.post("/clustering/silhouette")
async def clustering_silhouette(data: ClusteringData):
    X = np.array(data.X)
    labels = np.array(data.labels)
    fig = vis.plot_silhouette_scores(X, labels)
    path = save_figure(fig, "clustering_silhouette")
    return {"image_path": path}


@router.post("/clustering/cluster_centers")
async def clustering_cluster_centers(data: ClusteringData):
    if data.centers is None:
        raise HTTPException(status_code=400, detail="centers requis")
    X = np.array(data.X)
    labels = np.array(data.labels)
    centers = np.array(data.centers)
    fig = vis.plot_cluster_centers(X, labels, centers, method=data.method)
    path = save_figure(fig, "clustering_centers")
    return {"image_path": path}


@router.post("/clustering/cluster_distribution")
async def clustering_distribution(data: ClusteringData):
    labels = np.array(data.labels)
    fig = vis.plot_cluster_distribution(labels)
    path = save_figure(fig, "clustering_distribution")
    return {"image_path": path}


@router.post("/clustering/dendrogram")
async def clustering_dendrogram(data: ClusteringData):
    X = np.array(data.X)
    fig = vis.plot_dendrogram(X)
    path = save_figure(fig, "clustering_dendrogram")
    return {"image_path": path}


@router.get("/association/plot_association_rules_heatmap")
async def association_rules_heatmap(rules: str, metric: str = 'lift', top_n_graph: int = 20, top_n_itemsets: int = 10):
    """
    Expects `rules` as JSON string representing a DataFrame with columns:
    antecedents (list), consequents (list), support, confidence, lift
    Optionally, the itemsets can be loaded similarly from a JSON string or ignored here.
    """
    import json

    try:
        rules_df = pd.read_json(rules)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture règles JSON: {e}")

    figures = vis.plot_association_rules_heatmap(rules_df, metric=metric, top_n_graph=top_n_graph, top_n_itemsets=top_n_itemsets)

    saved_files = {}
    for name, fig in figures.items():
        filename = f"association_{name}.png"
        path = save_figure(fig, filename)
        saved_files[name] = path

    return saved_files