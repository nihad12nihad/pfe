# api/analyze_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from app.core.algorithms import get_algorithm
from app.core.visualization.results import (
    plot_classification_results,
    plot_regression_results,  # Supprimez cette ligne si vous ne faites pas de régression
    plot_confusion_matrix,
    plot_clusters
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
        raise HTTPException(status_code=500, detail="Erreur lors de l'analyse")

def generate_visualizations(result: dict, algorithm_name: str) -> Dict[str, str]:
    """Appelle les fonctions de visualisation appropriées"""
    vis_paths = {}
    
    # Visualisation des métriques  principales
    if 'metrics' in result:
        if 'accuracy' in result['metrics']:  # Classification
            vis_paths['metrics'] = plot_classification_results(
                result['metrics'],
                algorithm_name
            )
        elif 'r2_score' in result['metrics']:  # Régression
            vis_paths['metrics'] = plot_regression_results(
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
        vis_paths['clusters'] = plot_clusters(
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