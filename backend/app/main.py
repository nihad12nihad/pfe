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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    media_dir = Path("media")
    media_dir.mkdir(exist_ok=True)
    
    yield
    
    await async_engine.dispose()
    
app = FastAPI(
    lifespan=lifespan,
    title="Data Mining Platform",
    description="Plateforme d'analyse de données",
    redirect_slashes=True,
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

from fastapi import FastAPI,Request,Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.auth_routes import router as auth_router
from app.api import dataset_routes
from app.config.database import async_engine
from app.core.models import Base
from pathlib import Path
from app.api.dataset_routes import api_router as dataset_api_router
from app.api.dataset_routes import ui_router as dataset_ui_router
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.auth.security import get_current_user
from app.core.models import Utilisateur


BASE_DIR=Path(__file__).parent
templates=Jinja2Templates(directory=str(BASE_DIR.parent / "templates"))
app.mount("/static", StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="templates")

app.include_router(auth_router, prefix="/auth")

@app.on_event("startup")
async def startup_db():
    Base.metadata.create_all(bind=async_engine)

    
app.include_router(dataset_api_router)
app.include_router(dataset_ui_router)

@app.get("/", response_class=HTMLResponse)
async def root(request:Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request}
    )

@app.get("/accueil", response_class=HTMLResponse)
async def home (request: Request):
    return templates.TemplateResponse(
        request,
        "accueil.html",
        {"request":request}
    )

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

