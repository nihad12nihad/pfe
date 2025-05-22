from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.api import (
    analyze_routes,
    export_routes,
    upload_routes,
    preprocess_routes,
    info_routes,
    result_visualisation_routes,
    visualisation_routes,
    compare_routes,
    auth_routes
)

from app.config.database import engine
from app.core.models import Base

app = FastAPI(
    title="Data Mining Platform",
    description="Plateforme d'analyse de données",
    version="1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include all routes
app.include_router(auth_routes.router, prefix="/auth", tags=["Authentication"])
app.include_router(analyze_routes.router, prefix="/api/analyze", tags=["Analysis"])
app.include_router(export_routes.router, prefix="/api/export", tags=["Export"])
app.include_router(upload_routes.router, prefix="/api/upload", tags=["Upload"])
app.include_router(preprocess_routes.router, prefix="/api/preprocess", tags=["Preprocessing"])
app.include_router(info_routes.router, prefix="/api/info", tags=["Information"])
app.include_router(visualisation_routes.router, prefix="/api/visualize", tags=["Visualization"])
app.include_router(result_visualisation_routes.router, prefix="/api/results", tags=["Results"])
app.include_router(compare_routes.router, prefix="/api/compare", tags=["Comparison"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Mining Platform API"}

@app.on_event("startup")
async def startup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}