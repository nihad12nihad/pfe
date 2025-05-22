from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from app.core.visualization.charts import (
    plot_confusion_matrix,
    plot_classification_results,
    plot_regression_results,
    plot_clustering_results,
    plot_association_rules
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.post("/visualisation-resultats", response_class=HTMLResponse)
async def visualisation_resultats(
    request: Request,
    graph_type: str = Form(...),
    model_output_path: str = Form(...),
):
    try:
        # Lecture du fichier contenant les résultats du modèle
        df = pd.read_csv(model_output_path)  # Exemple : prédictions ou données transformées

        # Dictionnaire associant chaque type de graphique à sa fonction de génération
        plot_handlers = {
            "confusion": lambda: plot_confusion_matrix(df),
            "classification": lambda: plot_classification_results(df['y_true'], df['y_pred']),
            "regression": lambda: plot_regression_results(df['y_true'], df['y_pred']),
            "clustering": lambda: plot_clustering_results(df),
            "association": lambda: plot_association_rules(df),
        }

        if graph_type not in plot_handlers:
            raise ValueError("Type de graphique non supporté")

        image_path = plot_handlers[graph_type]()

        return templates.TemplateResponse("visualisation_resultats.html", {
            "request": request,
            "image_path": image_path.replace('backend/', '/')
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": str(e)
        })
