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