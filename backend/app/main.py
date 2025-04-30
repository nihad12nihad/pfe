from fastapi import FastAPI
from app.api import (
    auth_routes,
    upload_routes,
    preprocess_routes,
    analyze_routes,
    compare_routes,
    export_routes
)

app = FastAPI(title="Data Mining API", version="1.0")

# Inclusion des routes depuis les différents modules
app.include_router(auth_routes.router, prefix="/auth", tags=["Auth"])
app.include_router(upload_routes.router, prefix="/upload", tags=["Upload"])
app.include_router(preprocess_routes.router, prefix="/preprocess", tags=["Preprocessing"])
app.include_router(analyze_routes.router, prefix="/analyze", tags=["Analyze"])
app.include_router(compare_routes.router, prefix="/compare", tags=["Compare"])
app.include_router(export_routes.router, prefix="/export", tags=["Export"])

# Route de test
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de fouille de données 🎯"}



