<<<<<<< HEAD
DATABSES = {
    'default': {
        'ENGINE':'django.db.backends.mysql',
        'NAME': 'django_user',
        'USER': 'root',
        'PASSWORD': 'raouf124',
        'HOST': 'localhost',
        'PORT': '3306',
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
        }
        
    }
    
}

from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DB_USER: str = "root"
    DB_PASSWORD: str = "raouf124"
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_NAME: str = "django_user"
    MEDIA_ROOT: str = str(Path(__file__).parent.parent / "media")
    SECRET_KEY: str = "votre_secret_complexe_ici"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
Settings = Settings()
=======
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
>>>>>>> f440206c5d1d41dcbf0adf5979fddaaaa5776bbb
