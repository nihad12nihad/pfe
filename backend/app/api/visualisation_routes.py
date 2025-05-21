from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
import uuid
from app.core.visualization import charts

router = APIRouter()

OUTPUT_DIR = "app/data/resultats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_figure(fig, prefix="plot"):
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    fig.clf()
    return filepath

@router.post("/visualization/correlation_matrix")
async def correlation_matrix(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        fig = charts.plot_correlation_matrix(df)
        path = save_figure(fig, "correlation")
        return JSONResponse(content={"image_path": path})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/visualization/histogram")
async def histogram(file: UploadFile = File(...), column: str = ""):
    try:
        df = pd.read_csv(file.file)
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Colonne '{column}' non trouvée")
        fig = charts.plot_histogram(df, column)
        path = save_figure(fig, "histogram")
        return JSONResponse(content={"image_path": path})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/visualization/boxplot")
async def boxplot(file: UploadFile = File(...), column: str = ""):
    try:
        df = pd.read_csv(file.file)
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Colonne '{column}' non trouvée")
        fig = charts.plot_boxplot(df, column)
        path = save_figure(fig, "boxplot")
        return JSONResponse(content={"image_path": path})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/visualization/missing_values")
async def missing_values(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        fig = charts.plot_missing_values(df)
        if fig is None:
            return JSONResponse(content={"message": "Pas de valeurs manquantes détectées"})
        path = save_figure(fig, "missing_values")
        return JSONResponse(content={"image_path": path})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/visualization/unique_values")
async def unique_values(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        fig = charts.plot_unique_values(df)
        path = save_figure(fig, "unique_values")
        return JSONResponse(content={"image_path": path})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
