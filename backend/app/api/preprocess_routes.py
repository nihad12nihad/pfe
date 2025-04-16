from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

router = APIRouter(tags=["Preprocessing"])
logging.basicConfig(filename='preprocessing.log', level=logging.INFO)

# -------------------------------------------------------------------------
# 1. Gestion Intelligente des Valeurs Manquantes
# -------------------------------------------------------------------------
@router.post("/handle_missing")
async def handle_missing_values(
    filepath: str,
    strategy: str = "auto",
    custom_values: Optional[Dict[str, str]] = None,
    threshold: float = 0.05
):
    """
    Args:
        filepath: Chemin du fichier CSV
        strategy: "auto"|"drop"|"mean"|"median"|"mode"|"custom"
        custom_values: {"col1": "valeur", "col2": "moyenne"} (si strategy="custom")
        threshold: Seuil (0-1) pour suppression auto (ex: 0.05 = 5%)
    Returns:
        {"processed_file": "chemin_du_fichier_traite.csv", "stats": {"lignes_supprimees": 10, ...}}
    """
    try:
        # Chargement des données
        df = pd.read_csv(filepath)
        original_rows = len(df)
        stats = {"lignes_originales": original_rows}

        # Détection des NaN
        nan_counts = df.isna().sum()
        stats["nan_par_colonne"] = nan_counts.to_dict()

        # Stratégie Automatique (Recommandée)
        if strategy == "auto":
            for col in df.columns:
                # Suppression si NaN < seuil ET colonne critique
                if (nan_counts[col] / original_rows < threshold) and (df[col].dtype in ['int64', 'float64']):
                    df.dropna(subset=[col], inplace=True)
                    logging.info(f"Suppression des NaN dans {col} (<{threshold*100}%)")
                # Imputation sinon
                else:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)

        # Stratégie Manuelle
        elif strategy == "drop":
            df.dropna(inplace=True)
        elif strategy == "mean":
            df.fillna(df.mean(numeric_only=True), inplace=True)
        elif strategy == "median":
            df.fillna(df.median(numeric_only=True), inplace=True)
        elif strategy == "mode":
            df.fillna(df.mode().iloc[0], inplace=True)
        elif strategy == "custom" and custom_values:
            for col, val in custom_values.items():
                if val == "moyenne":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif val == "mediane":
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(val, inplace=True)

        # Sauvegarde et logs
        processed_path = filepath.replace(".csv", "_processed.csv")
        df.to_csv(processed_path, index=False)
        
        stats.update({
            "lignes_restantes": len(df),
            "lignes_supprimees": original_rows - len(df),
            "colonnes_modifiees": nan_counts[nan_counts > 0].index.tolist()
        })
        logging.info(f"Traitement terminé. Stats: {stats}")

        return {
            "processed_file": processed_path,
            "stats": stats
        }

    except Exception as e:
        logging.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")git --version
