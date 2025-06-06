import pandas as pd
import json
import pickle
import os
from typing import Dict, Any, Optional
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

def ensure_dir_exists(output_dir: str) -> Path:
    """Crée le dossier s'il n'existe pas et retourne le chemin absolu"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)  # Création de toute l'arborescence si nécessaire
    return path

def export_to_csv(data: dict, filename: str, output_dir: str = "app/data/resultats") -> str:
    """Exporte un dictionnaire en CSV"""
    dir_path = ensure_dir_exists(output_dir)
    filepath = dir_path / f"{filename}.csv"
    
    # Convertir le dictionnaire en DataFrame et exporter
    pd.DataFrame(data.items(), columns=["Metric", "Value"]).to_csv(filepath, index=False)
    return str(filepath)

def export_to_json(data: dict, filename: str, output_dir: str = "app/data/resultats") -> str:
    """Exporte un dictionnaire en JSON"""
    dir_path = ensure_dir_exists(output_dir)
    filepath = dir_path / f"{filename}.json"
    
    # Exporter le dictionnaire au format JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return str(filepath)

def export_to_excel(data: dict, filename: str, output_dir: str = "app/data/resultats") -> str:
    """Exporte un dictionnaire en Excel"""
    dir_path = ensure_dir_exists(output_dir)
    filepath = dir_path / f"{filename}.xlsx"
    
    # Convertir le dictionnaire en DataFrame et exporter
    pd.DataFrame(data.items(), columns=["Metric", "Value"]).to_excel(filepath, index=False, engine='openpyxl')
    return str(filepath)
