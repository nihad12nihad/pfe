# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.auth import routes as routes
from app.api import (
    analyze_routes,
    export_routes,
    upload_routes,
    preprocess_routes,
    info_routes,
    result_visualisation_routes,
    visualisation_routes
)


app = FastAPI(
    title="Data Mining Platform",
    description="Plateforme d'analyse de données",
    version="1.0"
)

# ✅ Ajouter le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # autorise le frontend React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routes
app.include_router(analyze_routes.router)
app.include_router(export_routes.router)
app.include_router(upload_routes.router)
app.include_router(preprocess_routes.router)
app.include_router(info_routes.router)
app.include_router(visualisation_routes.router)
app.include_router(result_visualisation_routes.router)


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur la plateforme de data mining"}