from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from pydantic import BaseModel, ConfigDict

Base = declarative_base()

class Utilisateur(Base):
    __tablename__ = "utilisateurs"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String(50), unique=True)
    mot_de_passe = Column(String(255))
    datasets = relationship("Dataset", back_populates="utilisateur")
    

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String(255))
    description = Column(Text)
    chemin = Column(String(512))
    taille = Column(Integer)
    date_upload = Column(DateTime, default=datetime.utcnow)
    utilisateur_id = Column(Integer, ForeignKey("utilisateurs.id"))
    
    utilisateur = relationship("Utilisateur", back_populates="datasets")
    
class DatasetSchema(BaseModel):
    id:int
    nom:str
    description:str
    chemin:str
    taille:int
    date_upload:datetime
    utilisateur:int
    
    model_config = ConfigDict(from_attributes=True)