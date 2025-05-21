from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    username: str
    password1: str = Field(min_length=6)
    password2: str = Field(min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class UserPublic(BaseModel):
    id: int
    username: str

    class Config:
        orm_mode = True

class UserPublic(BaseModel):
    id: int
    username: str

    class Config:
        orm_mode = True
