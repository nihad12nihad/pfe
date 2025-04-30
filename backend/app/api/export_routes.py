from fastapi import APIRouter, HTTPException, Depends, status, Security
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from pydantic import BaseModel, validator, Field
from pathlib import Path
from datetime import datetime
import logging
import os
from typing import Dict,Literal,Optional
import shutil

# Import de vos fonctions existantes
from results.export import (
    export_to_csv,
    export_to_json,
    export_to_excel,
    ensure_dir_exists
)

# définit le préfixe /api/export pour toutes les routes
router = APIRouter(
    prefix="/api/export",
    tags=["Export"],
    responses={404: {"description": "Not found"}}
)

logger = logging.getLogger(__name__)
EXPORT_DIR = Path("backend/data/processed").resolve()
security=HTTPBearer()

class ExportRequest(BaseModel):
    """Modèle de validation pour les exports"""
    metrics: Dict[str, float]
    filename: str = Field(default_factory=lambda: f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",examples="model_metrics",max_length=50)
    format: Literal["csv","json","excel"]="csv"

    @validator('filename')
    def validate_filename(cls, v):
        """Valide que le nom de fichier est sûr"""
        if not all(c.isalnum() or c in {'_', '-'} for c in v):
            raise ValueError("Caractères non autorisés dans le nom de fichier")
        if len(v) > 50:
            raise ValueError("Le nom de fichier est trop long (max 50 caractères)")
        return v


class ExportResponse(BaseModel):
    status: Literal["success","error"]
    path: Optional[str]=None
    download_url: Optional[str]=None
    size: Optional[str]=None
    error: Optional[str]=None
    

def get_secure_path(filename: str) -> Path:
    """Valide et sécurise le chemin de sortie"""
    clean_name= Path(filename).name
    path=(EXPORT_DIR / clean_name).resolve()
    
    if not path.parent.samefile(EXPORT_DIR):
        logger.warning(f"Tentative d'accès non sécursé :{filename}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="accès refusé : chemin invalide"
        )
    return path

def cleanup_old_exports(max_files: int = 50):
    try:
        files = sorted(EXPORT_DIR.iterdir(), key=os.path.getmtime)
        if len(files) > max_files:
            for old_files in files[:len(files)-max_files]:
                old_files.unlink()
    except Exception as e:
        logger.warning(f"echec de nettoyage des exports : {e}")

@router.post(
    "/{export_format}",
    response_model=ExportResponse,
    responses={
        201:{"description":"fichier créé","model": ExportResponse},
        400:{"description":"requete invalide"},
        403:{"description":"accés refusé"},
        500:{"description":"erreur serveur"}
    }
)
async def export_data(
    export_format: Literal["csv","json","excel"],
    request: ExportRequest,
    credentials: HTTPAuthorizationCredentials=Security(security)
):
    
    """
    Exporte des métriques dans le format specifié
    
     requête :
     -**metrics**:dictionnaire de métriques
     -**filename**:nom de fichier(optionnel)
     -**format**:format d'export(optionnel, défaut: csv)
    """
    try:
        # Validation
        safe_filename = f"{request.filename}.{export_format}"
        output_path = get_secure_path(safe_filename)
        
        #dispatch selon le format
        exporters={
            "csv": export_to_csv,
            "json": export_to_json,
            "excel": export_to_excel
        }
        
        filepath = exporters[export_format](
            data=request.metrics,
            filename=request.filename,
            output_dir=str(EXPORT_DIR)  # Respecte votre signature
        )
        
        # Vérification que le fichier a bien été créé
        if not Path(filepath).exists():
            raise RuntimeError("echec de la creation du fichier")
        
        cleanup_old_exports()
            
        return ExportResponse(
            status= "succes",
            path=str(Path(filepath).relative_to(EXPORT_DIR)),
            download_url=f"/api/export/download/{safe_filename}",
            size=f"{os.path.getsize(filepath) / 1024:.2f} KB"
        )
        
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        
    except Exception as e:
        logger.error(f"Erreur d'export {export_format}: {str(e)}",exc_info=True)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la génération du {export_format}"
        )

@router.get(
    "/download/{filename}",
    response_class=FileResponse,
    summary="Télécharger un export",
    responses={
        200: {"description": "Fichier téléchargé"},
        403: {"description": "Accès refusé"},
        404: {"description": "Fichier non trouvé"}
    }
)
async def download_export_file(
    filename: str,
    credentials: HTTPAuthorizationCredentials=Security(security)
):
    """
    Télécharge un fichier exporté précédemment.
    Protection contre les attaques par traversal path.
    """
    try:
        filepath = get_secure_path(filename)
        
        if not filepath.exists():
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail="Fichier non trouvé"
            )
            
        return FileResponse(
            filepath,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur de téléchargement : {str(e)}", exc_info=True)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors du téléchargement"
        )