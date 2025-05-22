from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth.security import get_current_user
from app.config.database import get_db
from app.core.models import Utilisateur, Dataset, DatasetSchema
from app.config.settings import Settings
from pathlib import Path
import os 
import uuid
from fastapi.templating import Jinja2Templates

api_router = APIRouter(
    prefix="/api/datasets",
    tags=["Datasets API"]
)
ui_router = APIRouter(prefix="/datasets", tags=["Datasets UI"])

templates = Jinja2Templates(directory="templates")

@api_router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    description: str ="Aucune description",
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.lower().endswith(('.csv','.arff')):
        raise HTTPException(400, detail="Seuls les formats CSV/ARFF sont autorisés")
    
    user_dir = Path(Settings.MEDIA_ROOT) / f"user_{current_user.id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    
    file_ext = Path(file.filename).suffix
    unique_name = f"{uuid.uuid4()}{file_ext}"
    file_path = user_dir / unique_name
    
    contents = await file.read()
    try :
        with open(file_path,"wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur d'upload: {str(e)}")
    
    try :
        dataset = Dataset(
            nom=file.filename,
            description=description,
            chemin=str(file_path.relative_to(Settings.MEDIA_ROOT)),
            utilisateur_id=current_user.id,
            taille=len(contents)
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, detail=f"Erreur base de données: {str(e)}")
    
    return RedirectResponse("/datasets", status_code=status.HTTP_303_SEE_OTHER)

@api_router.get("", response_model=list[DatasetSchema])
async def list_datasets(
    current_user: Utilisateur = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    results = await db.execute(
        select(Dataset)
        .where(Dataset.utilisateur_id == current_user.id)
        .order_by(Dataset.date_upload.desc())
    )
    return results.scalars().all()

@api_router.get("/download/{dataset_id}")
async def download_dataset(
    dataset_id: int,
    current_user: Utilisateur = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Dataset)
        .where(Dataset.id == dataset_id)
        .where(Dataset.utilisateur_id == current_user.id)
    )
    dataset = result.scalars().first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset non trouvé"
        )
        
    full_path = Path(Settings.MEDIA_ROOT) / dataset.chemin
    if not full_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fichier physique introuvable"
        )
        
    return FileResponse(
        full_path,
        filename=dataset.nom,
        media_type="application/octet-stream"
    )

@ui_router.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse(
        request,
        "upload.html",
        {"request": request}
    )

    
@ui_router.get("", response_class=HTMLResponse)
async def list_datasets_ui(
    request: Request,
    current_user: Utilisateur=Depends(get_current_user),
    db: AsyncSession=Depends(get_db)
):
    results=await db.execute(
        select(Dataset)
         .where(Dataset.utilisateur_id==current_user.id)
         .order_by(Dataset.date_upload.desc())
         )
    datasets = results.scalars().all()
    return templates.TemplateResponse(
        request,
         "datasets.html", 
         {"request": request, "datasets": datasets}
         )
    
__all__ = ["api_router", "ui_router"]
    
    