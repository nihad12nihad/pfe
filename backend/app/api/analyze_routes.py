# api/analyze_routes.py
# api/analyze_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import uuid
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.core.algorithms import get_algorithm
<<<<<<< HEAD
from app.core.visualization.results import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_classification_scores,
    plot_feature_importances,
    plot_residuals,
    plot_true_vs_pred,
    plot_regression_errors,
    plot_clusters_2d,
    plot_silhouette,
    plot_dendrogram,
    plot_association_rules_graph,
    display_rules_table
)
from app.results.export import export_to_csv, export_to_json, export_to_excel
from app.core.algorithms import get_algorithm
from app.core.visualization.results import (
    plot_classification_scores,
    plot_regression_errors,  # Supprimez cette ligne si vous ne faites pas de régression
    plot_confusion_matrix,
    plot_clusters_2d
)
=======
from app.core.visualization import results as vis  # import tout le module pour centraliser
>>>>>>> f440206c5d1d41dcbf0adf5979fddaaaa5776bbb
from app.results.export import export_to_csv, export_to_json, export_to_excel

router = APIRouter()
logger = logging.getLogger("ML_API")

OUTPUT_DIR = "app/data/resultats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_figure(fig, prefix: str) -> str:
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path

class AnalyzeRequest(BaseModel):
    algorithm: str
    data_path: str
    target_column: Optional[str] = None
    parameters: Dict[str, Any] = {}
    export_format: Optional[str] = None  # 'csv', 'json', 'excel'

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    try:
        algo_func = get_algorithm(request.algorithm)
        result = algo_func(
            data_path=request.data_path,
            target_column=request.target_column,
            **request.parameters
        )

        vis_paths = generate_visualizations(result, request.algorithm)

        export_path = None
        if request.export_format:
            export_path = export_results(
                result.get('metrics', {}),
                request.algorithm,
                request.export_format
            )

        return {
            "status": "success",
            "algorithm": request.algorithm,
            "metrics": result.get('metrics', {}),
            "visualizations": vis_paths,
            "export_path": export_path
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erreur serveur: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def generate_visualizations(result: dict, algorithm_name: str) -> Dict[str, str]:
    vis_paths = {}

    metrics = result.get('metrics', {})
    # Classification (exemple detection par présence de 'accuracy')
    if 'accuracy' in metrics:
        cm = result.get('confusion_matrix')
        if cm is not None:
            fig = vis.plot_confusion_matrix(cm, labels=result.get('class_labels', []))
            vis_paths['confusion_matrix'] = save_figure(fig, "confusion_matrix")

        y_true = result.get('actual')
        y_pred = result.get('predictions')
        y_scores = result.get('scores')  # scores probas pour ROC/PR si dispo
        if y_true is not None and y_scores is not None:
            fig_roc = vis.plot_roc_curve(y_true, y_scores)
            vis_paths['roc_curve'] = save_figure(fig_roc, "roc_curve")
            fig_pr = vis.plot_precision_recall_curve(y_true, y_scores)
            vis_paths['precision_recall'] = save_figure(fig_pr, "precision_recall")

        # Feature importance (optionnel)
        feature_importances = result.get('feature_importances')
        feature_names = result.get('feature_names')
        if feature_importances is not None and feature_names is not None:
            class DummyModel:
                feature_importances_ = feature_importances
            fig_fi = vis.plot_feature_importance(DummyModel(), feature_names)
            vis_paths['feature_importance'] = save_figure(fig_fi, "feature_importance")

        # Decision tree visualization (optionnel)
        model = result.get('model')
        class_names = result.get('class_labels')
        if model is not None and class_names is not None:
            fig_dt = vis.plot_decision_tree(model, result.get('feature_names', []), class_names)
            vis_paths['decision_tree'] = save_figure(fig_dt, "decision_tree")

    # Régression (exemple détection par présence de r2_score)
    elif 'r2_score' in metrics:
        y_true = result.get('actual')
        y_pred = result.get('predictions')
        if y_true is not None and y_pred is not None:
            fig_reg = vis.plot_regression_results(y_true, y_pred)
            vis_paths['regression_results'] = save_figure(fig_reg, "regression_results")

            fig_err = vis.plot_regression_errors(y_true, y_pred)
            vis_paths['regression_errors'] = save_figure(fig_err, "regression_errors")

            coeffs = result.get('coefficients')
            intercept = result.get('intercept')
            if coeffs is not None:
                fig_coef = vis.plot_regression_coefficients(coeffs, intercept)
                vis_paths['regression_coefficients'] = save_figure(fig_coef, "regression_coefficients")

            fig_resid = vis.plot_residuals(y_true, y_pred)
            vis_paths['regression_residuals'] = save_figure(fig_resid, "regression_residuals")

            # 3D visualization optionnelle
            X_test = result.get('X_test')
            feature_names = result.get('feature_names')
            if X_test is not None and feature_names is not None:
                X_df = pd.DataFrame(X_test)
                fig_3d = vis.plot_3d_regression(X_df, y_true, y_pred, feature_names)
                vis_paths['regression_3d'] = save_figure(fig_3d, "regression_3d")

    # Clustering
    if 'cluster_labels' in result and 'visualization_data' in result:
        X = result['visualization_data']
        labels = result['cluster_labels']
        fig_clusters = vis.plot_clusters_2d(X, labels, method=result.get('method', 'pca'))
        vis_paths['clusters_2d'] = save_figure(fig_clusters, "clusters_2d")

        fig_sil = vis.plot_silhouette_scores(X, labels)
        vis_paths['silhouette_scores'] = save_figure(fig_sil, "silhouette_scores")

        centers = result.get('centers')
        if centers is not None:
            fig_centers = vis.plot_cluster_centers(X, labels, centers, method=result.get('method', 'pca'))
            vis_paths['cluster_centers'] = save_figure(fig_centers, "cluster_centers")

        fig_dist = vis.plot_cluster_distribution(labels)
        vis_paths['cluster_distribution'] = save_figure(fig_dist, "cluster_distribution")

        if 'dendrogram' in result:
            fig_dendro = vis.plot_dendrogram(result['dendrogram'])
            vis_paths['dendrogram'] = save_figure(fig_dendro, "dendrogram")

    # Association rules
    if 'association_rules' in result:
        # On suppose que association_rules est un DataFrame pandas
        fig_dict = vis.plot_association_rules_heatmap(
            result['association_rules'],
            metric=result.get('metric', 'lift'),
            top_n_graph=result.get('top_n_graph', 20),
            top_n_itemsets=result.get('top_n_itemsets', 10)
        )
        for name, fig in fig_dict.items():
            vis_paths[f"association_{name}"] = save_figure(fig, f"association_{name}")

    return vis_paths


def export_results(metrics: dict, algorithm_name: str, format: str) -> str:
    if format == 'csv':
        return export_to_csv(metrics, f"{algorithm_name}_results")
    elif format == 'json':
        return export_to_json(metrics, f"{algorithm_name}_results")
    elif format == 'excel':
        return export_to_excel(metrics, f"{algorithm_name}_results")
    else:
        raise ValueError(f"Format non supporté: {format}")
<<<<<<< HEAD

        logger.error(f"Erreur serveur: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de l'analyse")

def generate_visualizations(result: dict, algorithm_name: str) -> Dict[str, str]:
    """Appelle les fonctions de visualisation appropriées"""
    vis_paths = {}
    
    # Visualisation des métriques  principales
    if 'metrics' in result:
        if 'accuracy' in result['metrics']:  # Classification
            vis_paths['metrics'] = plot_classification_scores(
                result['metrics'],
                algorithm_name
            )
        elif 'r2_score' in result['metrics']:  # Régression
            vis_paths['metrics'] = plot_regression_errors(
                result['metrics'],
                algorithm_name
            )
    
    # Matrice de confusion pour la classification
    if 'confusion_matrix' in result:
        vis_paths['confusion_matrix'] = plot_confusion_matrix(
            result['confusion_matrix'],
            labels=result.get('class_labels', [])
        )
    
    # Visualisation des clusters
    if 'cluster_labels' in result:
        vis_paths['clusters'] = plot_clusters_2d(
            result.get('visualization_data', {}),
            algorithm_name
        )
    
    return vis_paths

def export_results(metrics: dict, algorithm_name: str, format: str) -> str:
    """Utilise les fonctions d'export de Personne 3"""
    if format == 'csv':
        return export_to_csv(metrics, f"{algorithm_name}_results")
    elif format == 'json':
        return export_to_json(metrics, f"{algorithm_name}_results")
    elif format == 'excel':
        return export_to_excel(metrics, f"{algorithm_name}_results")
    else:
        raise ValueError(f"Format non supporté: {format}")
=======
>>>>>>> f440206c5d1d41dcbf0adf5979fddaaaa5776bbb
