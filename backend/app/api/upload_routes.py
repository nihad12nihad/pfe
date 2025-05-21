#api/upload_routes.py
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from werkzeug.utils import secure_filename
from typing import Tuple, Optional
from pathlib import Path
import arff 

router = APIRouter()
# Configuration
UPLOAD_DIR = "app/data/raw"  # Chemin relatif depuis la racine du projet
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.txt', '.json', '.arff'}

def _save_uploaded_file(file: UploadFile) -> Tuple[Optional[str], Optional[str]]:
    """Sauvegarde le fichier uploadé et retourne son chemin"""
    try:
        filename = secure_filename(file.filename)
        if not filename:
            return None, "Nom de fichier invalide"
        
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return None, f"Format {ext} non supporté. Formats autorisés: {', '.join(ALLOWED_EXTENSIONS)}"
        
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        # Sauvegarde du fichier
        with open(filepath, "wb") as buffer:
            buffer.write(file.file.read())
            
        return filepath, None
        
    except Exception as e:
        return None, f"Erreur lors de l'enregistrement: {str(e)}"

def _convert_to_csv(filepath: str) -> Tuple[Optional[str], Optional[str]]:
    """Convertit le fichier en CSV si nécessaire"""
    try:
        ext = Path(filepath).suffix.lower()
        
        if ext == '.csv':
            return filepath, None  # Pas de conversion nécessaire
            
        # Lecture selon le format
        if ext in ('.xlsx', '.xls'):
            df = pd.read_excel(filepath)
        elif ext == '.txt':
            # Essaye plusieurs séparateurs courants
            try:
                df = pd.read_csv(filepath, sep=None, engine='python')  # Auto-détection
            except:
                return None, "Impossible de déterminer le séparateur du fichier TXT"
        elif ext == '.json':
            df = pd.read_json(filepath)
        elif ext == '.arff':
            with open(filepath, 'r', encoding='utf-8') as f:
                arff_data = arff.load(f)
            attributes = [attr[0] for attr in arff_data['attributes']]
            df = pd.DataFrame(arff_data['data'], columns=attributes)
        else:
            return None, "Format non implémenté pour la conversion"
        
        # Conversion en CSV
        csv_path = f"{os.path.splitext(filepath)[0]}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path, None
        
    except Exception as e:
        return None, f"Erreur de conversion: {str(e)}"

@router.post("/upload", response_class=JSONResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint pour uploader et convertir des fichiers de données
    Formats supportés: CSV, Excel (XLSX/XLS), TXT, JSON, ARFF
    """
    # 1. Sauvegarde du fichier original
    saved_path, error = _save_uploaded_file(file)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    # 2. Conversion en CSV si nécessaire
    csv_path, error = _convert_to_csv(saved_path)
    if error:
        # Supprime le fichier original en cas d'échec
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=400, detail=error)
    
    # 3. Réponse
    return {
        "status": "success",
        "original_path": saved_path,
        "csv_path": csv_path if csv_path != saved_path else None,
        "message": "Fichier traité avec succès"
    }