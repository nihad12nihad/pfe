from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api import preprocess_routes  # Correct
from app.api import visualisation_routes  # Utilisation de l'import relatif



# Création de l'application FastAPI
app = FastAPI()

# Inclure les routes de prétraitement
app.include_router(preprocess_routes.router, prefix="/api")

# Montée du dossier static pour que FastAPI puisse y accéder et servir les fichiers
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ajout des routes définies dans visualisation_routes.py
app.include_router(vis_router)


