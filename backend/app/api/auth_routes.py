from fastapi import APIRouter, Request,Depends,Form,status,HTTPException
from fastapi.responses import HTMLResponse,RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.config.database import get_db
from app.core.models import Utilisateur
from app.auth.security import hash_password,verify_password,pwd_context,create_access_token
#from app.main import templates
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter(tags=["Authentification"])

@router.get("/inscription", response_class=HTMLResponse)
async def inscription_page(request: Request):
    return templates.TemplateResponse(
        request,
        "inscription.html",
        {"request": request}
    )

@router.post("/inscription")
async def handle_inscription(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    try:
         if password != password_confirm:
             return templates.TemplateResponse(
                 "inscription.html", {
                 "request": request,
                 "error": "les mots de passe ne correspondent pas"
        })
             
         existing_user = await db.execute(
             select(Utilisateur).where(Utilisateur.nom== username)
             )
         if existing_user.scalars().first():
             return templates.TemplateResponse("inscription.html",{
                 "request": request,
                 "error": "Nom d'utilisteur déja utilisé"
        })
             
         hashed_password = hash_password(password)
         new_user = Utilisateur(nom=username, mot_de_passe=hashed_password)
         
         db.add(new_user)
         await db.commit()
         await db.refresh(new_user)
        
         return RedirectResponse(url="/auth/connexion", status_code=status.HTTP_303_SEE_OTHER)
      
    except Exception as e :
        await db.rollback()
        print(f"Erreur d'inscription : {str(e)}")
        return templates.TemplateResponse("inscription.html", {
            "request": request,
            "error": "Erreur technique lors de l'inscription"
        })

       
@router.get("/connexion", response_class=HTMLResponse)
async def connexion_page(request: Request):
    return templates.TemplateResponse(
        request,
        "connexion.html",
        {"request": request}
    )

@router.post("/connexion")
async def handle_connexion(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await db.execute(select(Utilisateur).where(Utilisateur.nom == username))
        user = result.scalars().first()
        
        if not user :
            return templates.TemplateResponse("connexion.html",{
                "request": request,
                "error_type": "user_not_found",
                "username": username
                })
            
        if not verify_password(password, user.mot_de_passe):
            return templates.TemplateResponse("connexion.html", {
                "request": request,
                "error_type": "wrong_password",
                "username": username
                })
            
        access_token = create_access_token({"sub": user.nom})
        response = RedirectResponse(url="/accueil", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True, secure=False, samesite="lax",max_age=3600)
        return response
    
    except Exception as e :
        print(f"ERREUR: {str(e)}")
        return templates.TemplateResponse("connexion.html", {
            "request": request,
            "error_type": "system_error"
        })
        
@router.get("/deconnexion")
async def deconnexion():
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")
    return response