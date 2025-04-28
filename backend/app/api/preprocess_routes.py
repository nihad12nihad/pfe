from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd

from app.core.preprocessing import handle_missing_values, encode_data, normalize_data, select_features

# Définir le router
router = APIRouter()

# Définir un modèle Pydantic pour valider les données entrantes
class PreprocessRequest(BaseModel):
    data: List[Dict[str, Any]]  # Liste de dictionnaires représentant les lignes du DataFrame

@router.post("/preprocess")
async def preprocess_data(request: PreprocessRequest):
    try:
        # 1. Convertir les données en DataFrame
        df = pd.DataFrame(request.data)

        # 2. Appliquer les traitements
        df = handle_missing_values(df)
        df = encode_data(df)
        df = normalize_data(df)
        df = select_features(df, target_col='maladie', method='kbest', k=3)

        # 3. Retourner les résultats sous forme de liste de dictionnaires
        return {"status": "success", "data": df.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
