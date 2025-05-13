from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import os

from app.core.preprocessing import (
    handle_missing_values,
    encode_data,
    normalize_data,
    select_features
)

router = APIRouter()

class PreprocessRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None  # facultatif si on donne un fichier
    filename: Optional[str] = None               # nouveau champ pour lire un fichier CSV
    steps: List[str] = []

    # Gestion des valeurs manquantes
    numeric_strategy: Optional[str] = 'mean'
    categorical_strategy: Optional[str] = 'constant'

    # Encodage
    encoding_strategy: Optional[str] = 'onehot'
    interval_cols: Optional[Dict[str, Any]] = None
    categorical_mapping: Optional[Dict[str, Dict[str, Any]]] = None

    # Normalisation
    normalization_method: Optional[str] = 'standard'
    normalization_columns: Optional[List[str]] = None

    # Sélection de variables
    selected_columns: Optional[List[str]] = None
    feature_selection_method: Optional[str] = 'kbest'
    target_col: Optional[str] = 'maladie'
    k: Optional[int] = 5
    percentile: Optional[int] = 20
    custom_features: Optional[List[str]] = None

@router.post("/preprocess")
async def preprocess_data(request: PreprocessRequest):
    try:
        # Étape 0 : Chargement des données depuis CSV ou JSON
        if request.filename:
            file_path = os.path.join("data/raw", request.filename)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"Fichier non trouvé : {file_path}")
            df = pd.read_csv(file_path)
        elif request.data:
            df = pd.DataFrame(request.data)
        else:
            raise HTTPException(status_code=400, detail="Aucune donnée fournie (ni fichier, ni données JSON).")

        # Étape 1 : Sélection manuelle si précisé
        if request.selected_columns:
            df = df[request.selected_columns + ([request.target_col] if request.target_col in df.columns else [])]

        # Étape 2 : Gestion des valeurs manquantes
        if 'handle_missing_values' in request.steps:
            df = handle_missing_values(
                df,
                numeric_strategy=request.numeric_strategy,
                categorical_strategy=request.categorical_strategy
            )

        # Étape 3 : Encodage
        if 'encode_data' in request.steps:
            df = encode_data(
                df,
                interval_cols=request.interval_cols,
                categorical_mapping=request.categorical_mapping,
                encoding_strategy=request.encoding_strategy
            )

        # Étape 4 : Normalisation
        if 'normalize_data' in request.steps:
            df = normalize_data(
                df,
                columns=request.normalization_columns,
                method=request.normalization_method
            )

        # Étape 5 : Sélection des variables
        if 'select_features' in request.steps:
            df = select_features(
                df,
                target_col=request.target_col,
                method=request.feature_selection_method,
                k=request.k,
                percentile=request.percentile,
                custom_features=request.custom_features
            )

        # Étape 6 : Sauvegarde dans data/processed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/processed/processed_{timestamp}.csv"
        df.to_csv(output_path, index=False)

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "file_path": output_path
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    