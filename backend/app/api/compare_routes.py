from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Literal,Optional
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import random
from pydantic import BaseModel

# Import de vos modules existants
#from app.core.algorithms import load_algorithm_results
from results.visualisation import plot_classification_results
from results.export import export_to_csv, export_to_json, export_to_excel

# def load_algorithm_results(algorithm_name: str) -> dict:
   #return {
       #"metrics":{
        #   "accuracy": 0.95,
         # "precision": 0.89
       #}
   #}
        

router = APIRouter(prefix="/compare", tags=["Algorithm Comparison"])
logger = logging.getLogger(__name__)

class ComparisonResponse(BaseModel):
    """Modèle de réponse pour la comparaison"""
    status: str
    plots: Dict[str, str]
    comparison_file: str
    file_format: str
    metrics_summary: Dict[str, Dict[str, float]]

def validate_algorithm_name(name: str) -> bool:
    """Valide la sécurité du nom d'algorithme"""
    return all(c.isalnum() or c in {'_', '-'} for c in name)

def mock_load_algorithm_results(algorithm_name: str) -> Dict[str,any]:
    allowed_chars= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-")
    if not all(c in allowed_chars for c in algorithm_name):
        raise ValueError(f"nom d'algorithme invalide : {algorithm_name}")
    
    base_metrics={
        "accuracy": round(random.uniform(0.8, 0.95), 4),
        "precision": round(random.uniform(0.75, 0.9), 4),
        "recall": round(random.uniform(0.7, 0.85), 4),
        "f2_score": round(random.uniform(0.72, 0.88), 4)
    }
    
    if "forest" in algorithm_name.lower():
        return {"metrics":{**base_metrics, "roc_auc": round(random.uniform(0.85, 0.98), 4)}}
    elif "svm" in algorithm_name.lower():
        return{"metrics": {**base_metrics, "hinge_loss": round(random.uniform(0.1, 0.3), 4)}}
    else:
        return {"metrics": base_metrics}
    
    

async def load_metrics_wrapper(algorithm_name: str) -> Dict[str, float]:
    """Charge les métriques d'un algorithme de manière asynchrone"""
    if not validate_algorithm_name(algorithm_name):
        raise ValueError(f"Nom d'algorithme invalide : {algorithm_name}")
    
    try:
        result = await asyncio.to_thread(mock_load_algorithm_results, algorithm_name)
        return result['metrics']
    except Exception as e:
        raise RuntimeError(f"erreur de chargement : {str(e)}")

@router.post(
    "/algorithms",
    response_model=ComparisonResponse,
    responses={
        200: {"description": "Comparaison réussie"},
        400: {"description": "Requête invalide"},
        500: {"description": "Erreur serveur"}
    }
)
async def compare_algorithms(
    algorithm_names: List[str],
    export_format: Literal["csv", "json", "excel"] = "csv"
) -> ComparisonResponse:
    """
    Compare plusieurs algorithmes et exporte les résultats
    
    Exemple de requête :
    {
        "algorithm_names": ["random_forest", "svm"],
        "export_format": "csv"
    }
    """
    # Validation des entrées
    if not algorithm_names:
        raise HTTPException(400, detail="La liste des algorithmes est vide")
    
    if len(set(algorithm_names)) != len(algorithm_names):
        raise HTTPException(400, detail="Doublons détectés dans les noms")

    # Chargement parallèle des résultats
    try:
        results = await asyncio.gather(*[load_metrics_wrapper(name) for name in algorithm_names])
    except Exception as e:
        logger.error(f"erreur de chargement : {str(e)}")
        raise HTTPException(500, detail=str(e))
    

    # Création des visualisations
    plots = {}
    for name, metrics in zip(algorithm_names, results):
        try:
            plot_path = plot_classification_results(metrics, name)
            plots[name] = str(plot_path)
        except Exception as e:
            logger.warning(f"Erreur de visualisation pour {name} : {str(e)}")
            plots[name]="error"
            
    export_data = {name: metrics for name, metrics in zip(algorithm_names,results)}
    
    try:
        filename = f"comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        if export_format == "csv":
            export_path = export_to_csv(export_data, filename)
        elif export_format =="json":
            export_path = export_to_json(export_data, filename)
        elif export_format == "excel":
            export_path = export_to_excel(export_data, filename)
        else:
            raise HTTPException(400, detail="format non supporté")
    except Exception as e :
        logger.error(f"errur d'export: {str(e)}")
        raise HTTPException(500, detail="echec de l'export des résultats")
    
    return ComparisonResponse(
        status="success",
        plots=plots,
        comparison_file=export_path,
        file_format=export_format,
        metrics_summary=export_data
    )