from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from app.core.algorithms import get_algorithm
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

router = APIRouter()
logger = logging.getLogger("ML_API")

class AnalyzeRequest(BaseModel):
    algorithm: str
    data_path: str
    target_column: Optional[str] = None
    parameters: Dict[str, Any] = {}
    export_format: Optional[str] = None  # 'csv', 'json', 'excel'

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    try:
        # 1. Récupérer et exécuter l'algorithme
        algo_func = get_algorithm(request.algorithm)
        result = algo_func(
            data_path=request.data_path,
            target_column=request.target_column,
            **request.parameters
        )
        
        # 2. Générer les visualisations adaptées au type d'algorithme
        vis_paths = generate_visualizations(result, request.algorithm)
        
        # 3. Exporter si demandé
        export_path = None
        if request.export_format:
            export_path = export_results(
                result['metrics'],
                request.algorithm,
                request.export_format
            )
        
        return {
            "status": "success",
            "algorithm": request.algorithm,
            "metrics": result['metrics'],
            "visualizations": vis_paths,
            "export_path": export_path
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erreur serveur: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def generate_visualizations(result: dict, algorithm_name: str) -> Dict[str, str]:
    """Appelle les fonctions de visualisation appropriées"""
    vis_paths = {}
    
    # Visualisation des métriques principales
    if 'metrics' in result:
        if 'accuracy' in result['metrics']:  # Classification
            vis_paths['metrics'] = plot_classification_scores(
                result['metrics'],
                algorithm_name
            )
            if 'confusion_matrix' in result:
                vis_paths['confusion_matrix'] = plot_confusion_matrix(
                    result['confusion_matrix'],
                    labels=result.get('class_labels', [])
                )
            if 'roc_auc' in result:
                vis_paths['roc_curve'] = plot_roc_curve(
                    result['actual'],
                    result['predictions']
                )

        elif 'r2_score' in result['metrics']:  # Régression
            vis_paths['metrics'] = plot_regression_errors(
                result['metrics'],
                algorithm_name
            )
            # ➕ Graphe réel vs prédit
            vis_paths['actual_vs_pred'] = plot_true_vs_pred(
                actual=result.get("actual", []),
                predicted=result.get("predictions", [])
            )

    # Visualisation des clusters (pour clustering)
    if 'cluster_labels' in result and 'visualization_data' in result:
        vis_paths['clusters'] = plot_clusters_2d(
            result['visualization_data'],
            result['cluster_labels']
        )
        vis_paths['silhouette'] = plot_silhouette(
            result['visualization_data'],
            result['cluster_labels']
        )
        vis_paths['dendrogram'] = plot_dendrogram(
            result['visualization_data']
        )

    # Visualisation des règles d'association (si applicable)
    if 'association_rules' in result:
        vis_paths['association_graph'] = plot_association_rules_graph(
            result['association_rules']
        )
        vis_paths['association_table'] = display_rules_table(
            result['association_rules']
        )

    return vis_paths


def export_results(metrics: dict, algorithm_name: str, format: str) -> str:
    """Utilise les fonctions d'export"""
    if format == 'csv':
        return export_to_csv(metrics, f"{algorithm_name}_results")
    elif format == 'json':
        return export_to_json(metrics, f"{algorithm_name}_results")
    elif format == 'excel':
        return export_to_excel(metrics, f"{algorithm_name}_results")
    else:
        raise ValueError(f"Format non supporté: {format}")
