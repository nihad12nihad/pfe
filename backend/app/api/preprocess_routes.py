from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
import os

from app.core.preprocessing import (
    handle_missing_values,
    encode_data,
    normalize_data,
    select_features
)

router = APIRouter()

class PreprocessRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None  # Facultatif si on donne un fichier
    filename: Optional[str] = None               # Nom du fichier CSV déjà uploadé
    steps: List[str] = []                        # Étapes à réaliser

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
        # Étape 0 : Chargement des données
        if request.filename:
            raw_dir = Path("app/data/raw")
            file_path = raw_dir / request.filename
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"Fichier non trouvé : {file_path}")
            ext = file_path.suffix.lower()
            if ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext == '.json':
                df = pd.read_json(file_path)
            else:
                raise HTTPException(status_code=400, detail=f"Format non supporté : {ext}")
        elif request.data:
            df = pd.DataFrame(request.data)
        else:
            raise HTTPException(status_code=400, detail="Aucune donnée fournie.")

        # Étape 1 : Sélection manuelle des colonnes
        if request.selected_columns:
            cols = list(request.selected_columns)
            if request.target_col in df.columns and request.target_col not in cols:
                cols.append(request.target_col)
            df = df[cols]

        # Vérifie que la colonne cible est présente
        if 'select_features' in request.steps and request.target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"La colonne cible '{request.target_col}' est absente des données.")

        # Étape 2 : Valeurs manquantes
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

        # Étape 6 : Sauvegarde
        processed_dir = Path("app/data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"processed_{timestamp}.csv"
        output_path = processed_dir / output_file
        df.to_csv(output_path, index=False)

        return {
            "status": "success",
            "message": "Prétraitement terminé avec succès.",
            "columns": df.columns.tolist(),
            "data": df.to_dict(orient="records"),
            "file_path": str(output_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prétraitement : {str(e)}")

@router.get("/preprocess/files")
async def list_processed_files():
    processed_dir = Path("app/data/processed")
    if not processed_dir.exists():
        raise HTTPException(status_code=404, detail="Dossier 'processed' introuvable.")
    files = [f.name for f in processed_dir.glob("*.csv")]
    return {"files": files}

@router.get("/preprocess/download/{filename}")
async def download_processed_file(filename: str):
    processed_dir = Path("app/data/processed")
    file_path = processed_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(path=file_path, filename=filename, media_type="text/csv")

@router.get("/preprocess/preview/{filename}")
async def preview_processed_file(filename: str, rows: int = 10):
    file_path = Path("app/data/processed") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    df = pd.read_csv(file_path)
    return {
        "columns": df.columns.tolist(),
        "preview": df.head(rows).to_dict(orient="records")
    }
