from fastapi import FastAPI

# Création de l'application FastAPI
app = FastAPI()

# Définition d'une route de test
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Point d'entrée du serveur
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
