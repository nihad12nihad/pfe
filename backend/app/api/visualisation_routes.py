from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from visualisation import (
    plot_correlation_matrix,
    plot_histogram,
    plot_boxplot,
    plot_categorical_count,
    plot_linear_relationship,
    plot_missing_values,
    plot_unique_values
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Route unique pour la visualisation avec sélection du graphique
@router.post("/visualisation", response_class=HTMLResponse)
async def visualisation(request: Request, file: UploadFile = File(...), graph_type: str = Form(...), column: str = Form(None), x_col: str = Form(None), y_col: str = Form(None)):
    df = pd.read_csv(file.file)

    # Choix du graphique
    if graph_type == "correlation":
        plot_correlation_matrix(df)
        image_path = "/static/graphique1.png"
    elif graph_type == "histogram":
        if not column:
            return {"error": "Veuillez spécifier une colonne pour l'histogramme."}
        plot_histogram(df, column)
        image_path = "/static/graphique2.png"
    elif graph_type == "boxplot":
        if not column:
            return {"error": "Veuillez spécifier une colonne pour le boxplot."}
        plot_boxplot(df, column)
        image_path = "/static/graphique3.png"
    elif graph_type == "countplot":
        if not column:
            return {"error": "Veuillez spécifier une colonne pour le countplot."}
        plot_categorical_count(df, column)
        image_path = "/static/graphique4.png"
    elif graph_type == "linear":
        if not x_col or not y_col:
            return {"error": "Veuillez spécifier les colonnes pour les axes X et Y."}
        plot_linear_relationship(df, x_col, y_col)
        image_path = "/static/graphique5.png"
    elif graph_type == "missing":
        plot_missing_values(df)
        image_path = "/static/graphique6.png"
    elif graph_type == "unique":
        plot_unique_values(df)
        image_path = "/static/graphique7.png"
    else:
        return {"error": "Type de graphique non reconnu"}

    # Renvoi de la réponse HTML avec le chemin de l'image générée
    return templates.TemplateResponse("visualisation.html", {
        "request": request,
        "image_path": image_path
    })
