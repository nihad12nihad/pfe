from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from app.core.visualization.preprocessing import(
    plot_correlation_matrix,
    plot_histogram,
    plot_boxplot,
    plot_missing_values,
    plot_unique_values
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.post("/visualisation", response_class=HTMLResponse)
async def visualisation(
    request: Request, 
    file: UploadFile = File(...), 
    graph_type: str = Form(...),
    column: str = Form(None),
    x_col: str = Form(None),
    y_col: str = Form(None)
):
    try:
        df = pd.read_csv(file.file)
        
        # Mapping des types de graphique
        plot_handlers = {
            "correlation": lambda: plot_correlation_matrix(df),
            "histogram": lambda: plot_histogram(df, column),
            "boxplot": lambda: plot_boxplot(df, column),
            "missing": lambda: plot_missing_values(df),
            "unique": lambda: plot_unique_values(df),
        }
        if graph_type not in plot_handlers:
            raise HTTPException(400, "Type de graphique non supporté")

        # Validation des paramètres
        if graph_type in ["histogram", "boxplot", "countplot"] and not column:
            raise HTTPException(400, f"Colonne requise pour {graph_type}")
        
        if graph_type == "linear" and (not x_col or not y_col):
            raise HTTPException(400, "x_col et y_col requis pour le graphique linéaire")

        # Génération du graphique
        image_path = plot_handlers[graph_type]()

        # Retour de l'image générée
        return templates.TemplateResponse("visualisation.html", {
            "request": request,
            "image_path": image_path.replace('backend/', '/')  # Pour le chemin web
        })

    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur interne: {str(e)}")

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from app.core.visualization.preprocessing import (
    plot_correlation_matrix,
    plot_histogram,
    plot_boxplot,
    plot_categorical_count,
    plot_linear_relationship,
    plot_missing_values,
    plot_unique_values
)
from app.core.visualization.results import (
    plot_confusion_matrix,
    plot_classification_scores
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.post("/visualisation", response_class=HTMLResponse)
async def visualisation(
    request: Request, 
    file: UploadFile = File(...), 
    graph_type: str = Form(...),
    column: str = Form(None),
    x_col: str = Form(None),
    y_col: str = Form(None)
):
    try:
        df = pd.read_csv(file.file)
        
        # Mapping des types de graphique
        plot_handlers = {
            "correlation": lambda: plot_correlation_matrix(df),
            "histogram": lambda: plot_histogram(df, column),
            "boxplot": lambda: plot_boxplot(df, column),
            "countplot": lambda: plot_categorical_count(df, column),
            "linear": lambda: plot_linear_relationship(df, x_col, y_col),
            "missing": lambda: plot_missing_values(df),
            "unique": lambda: plot_unique_values(df),
            #"confusion": lambda: plot_confusion_matrix(...),  # À adapter
            #"classification": lambda: plot_classification_results(...)  # À adapter
        }

        if graph_type not in plot_handlers:
            raise HTTPException(400, "Type de graphique non supporté")

        # Validation des paramètres
        if graph_type in ["histogram", "boxplot", "countplot"] and not column:
            raise HTTPException(400, f"Colonne requise pour {graph_type}")
        
        if graph_type == "linear" and (not x_col or not y_col):
            raise HTTPException(400, "x_col et y_col requis pour le graphique linéaire")

        # Génération du graphique
        image_path = plot_handlers[graph_type]()
        
        return templates.TemplateResponse("visualisation.html", {
            "request": request,
            "image_path": image_path.replace('backend/', '/')  # Pour le chemin web
        })

    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur interne: {str(e)}")