from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from app.config.settings import get_db
from app.auth import models, schemas, utils
from app.auth.dependencies import get_current_user

router = APIRouter()

@router.post("/register", response_model=schemas.UserPublic)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    if user.password1 != user.password2:
        raise HTTPException(status_code=400, detail="Les mots de passe ne correspondent pas")
    if db.query(models.User).filter(models.User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Nom d'utilisateur déjà pris")
    hashed = utils.get_password_hash(user.password1)
    db_user = models.User(username=user.username, hashed_password=hashed)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Nom d'utilisateur ou mot de passe incorrect")
    token = utils.create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@router.get("/dashboard", response_model=schemas.UserPublic)
def dashboard(current_user: models.User = Depends(get_current_user)):
    return current_user
