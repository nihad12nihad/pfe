<<<<<<< HEAD
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from visualisation import (
=======
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from core.visualization.preprocessing import (
>>>>>>> bfe5170 (par Nihadoun (nchlh ymchi khater ni bdit naeya))
    plot_correlation_matrix,
    plot_histogram,
    plot_boxplot,
    plot_categorical_count,
    plot_linear_relationship,
    plot_missing_values,
    plot_unique_values
)
<<<<<<< HEAD
=======
from core.visualization.results import (
    plot_confusion_matrix,
    plot_classification_results
)
>>>>>>> bfe5170 (par Nihadoun (nchlh ymchi khater ni bdit naeya))

router = APIRouter()
templates = Jinja2Templates(directory="templates")

<<<<<<< HEAD
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
=======
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
>>>>>>> bfe5170 (par Nihadoun (nchlh ymchi khater ni bdit naeya))
