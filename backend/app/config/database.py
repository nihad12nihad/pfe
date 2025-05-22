from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.config.settings import Settings

"""
DATABASE_URL = "mysql+asyncmy://root:raouf124@localhost/django_user"

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=True
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
        

# Format : mysql+pymysql://<user>:<password>@<host>/<dbname>
SQLALCHEMY_DATABASE_URL = f"mysql+asyncmy://{Settings.DB_USER}:{Settings.DB_PASSWORD}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    echo=False  # Mettre à True pour le débogage
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        """
        
# Configuration asynchrone uniquement
DATABASE_URL = (
    f"mysql+asyncmy://{Settings.DB_USER}:{Settings.DB_PASSWORD}"
    f"@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}"
)

async_engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=True
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session