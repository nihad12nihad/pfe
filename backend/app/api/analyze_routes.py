from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from core.algorithms import get_algorithm

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML_API")

class AnalyzeRequest(BaseModel):
    algorithm: str
    data_path: str
    target_column: Optional[str] = None
    parameters: Dict[str, Any] = {}

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    logger.info(f"Analyse demandée pour: {request.algorithm}")

    try:
        algo_function = get_algorithm(request.algorithm)

        kwargs = {"data_path": request.data_path}
        if request.target_column:
            kwargs["target_column"] = request.target_column
        kwargs.update(request.parameters)

        result = algo_function(**kwargs)

        logger.info(f"{request.algorithm} exécuté avec succès.")

        return {
            "status": "success",
            "algorithm": request.algorithm,
            "result": result
        }

    except ValueError as ve:
        logger.warning(f"Erreur utilisateur: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.error(f"Erreur lors de l'analyse : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

