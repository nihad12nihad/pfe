from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status,Request
from fastapi.security import OAuth2PasswordBearer,HTTPBearer,HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config.settings import Settings
from app.config.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.models import Utilisateur

pwd_context = CryptContext(schemes=["bcrypt"], bcrypt__default_rounds=12)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password,hashed_password)

def create_access_token(data: dict) -> str:
    return jwt.encode(
        data,
        Settings.SECRET_KEY,
        algorithm=Settings.ALGORITHM
    )
    
bearer_scheme = HTTPBearer(auto_error=False)

async def get_token_from_cookie_or_header(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)
) -> str | None:
    token = request.cookies.get("access_token")
    if token:
        return token
    
    if creds and creds.scheme.lower() == "bearer":
        return creds.credentials
    
    return None


#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/connexion")


async def get_current_user(
    token: str | None = Depends(get_token_from_cookie_or_header),
    db: AsyncSession = Depends(get_db)
) -> Utilisateur:
    if not token:
        raise HTTPException(
             status_code=status.HTTP_401_UNAUTHORIZED,
             detail="non authentifié",
             headers={"WWW-Authenticate": "Bearer"},          
        )
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré",
        headers={"WWW-Authenticate":"Bearer"},
    )
    
    
    try:
        raw = token.split(" ")[-1]
        payload = jwt.decode(
            raw,
            Settings.SECRET_KEY,
            algorithms=[Settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    result = await db.execute(select(Utilisateur).where(Utilisateur.nom == username))
    user = result.scalars().first()
    
    if user is None:
        raise credentials_exception
        
    return user