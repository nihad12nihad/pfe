from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
from typing import Optional

router = APIRouter()

DATA_DIR = "app/data/raw"

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Tente de détecter une colonne cible (ex: 'target', 'label', etc.)"""
    candidates = ['target', 'label', 'class', 'outcome']
    for col in df.columns:
        if col.strip().lower() in candidates:
            return col
    return None

@router.get("/dataset/info", response_class=JSONResponse)
def get_dataset_info(
    filename: str = Query(..., description="Nom du fichier à analyser"),
    sep: str = Query(",", description="Séparateur CSV (par défaut: ',')")
):
    filepath = os.path.join(DATA_DIR, filename)

    # Vérifie que le fichier est bien un type supporté
    if not filename.lower().endswith(('.csv', '.json', '.xlsx', '.arff')):
        raise HTTPException(status_code=400, detail="Type de fichier non supporté")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Fichier introuvable")

    try:
        # Tentative de lecture avec encodage par défaut puis fallback
        try:
            df = pd.read_csv(filepath, sep=sep)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, sep=sep, encoding='latin-1')

        # Infos de base
        n_rows, n_cols = df.shape
        head_preview = df.head(5).fillna("").to_dict(orient='records')

        # Détection de la colonne cible
        target_col = detect_target_column(df)

        # Valeurs manquantes
        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        missing_suggestions = {}

        for col, count in missing_dict.items():
            dtype = str(df[col].dtype)
            if "float" in dtype or "int" in dtype:
                suggestion = "remplir par moyenne/médiane ou supprimer"
            else:
                suggestion = "remplir par mode ou supprimer"
            missing_suggestions[col] = {
                "missing_count": int(count),
                "suggestion": suggestion
            }

        # Statistiques globales
        numeric_df = df.select_dtypes(include=['number'])
        stats = numeric_df.describe().round(3).to_dict()
        median = numeric_df.median().round(3).to_dict()

        return JSONResponse(content={
            "status": "success",
            "shape": {"rows": n_rows, "columns": n_cols},
            "preview": head_preview,
            "target_column": target_col,
            "missing_clean": len(missing_dict) == 0,
            "missing_values": missing_suggestions if missing_suggestions else "Aucune valeur manquante détectée ✅",
            "statistics": stats,
            "median": median
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse du fichier : {str(e)}")
