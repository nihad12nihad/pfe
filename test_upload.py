from werkzeug.datastructures import FileStorage
from backend.app.api.upload_routes import upload_file
import pandas as pd
import os

try:
    import openpyxl
except ImportError:
    print("Erreur: Le module 'openpyxl' n'est pas installé.")
    print("Installez-le avec : pip install openpyxl")
    exit(1)


# 1. Créer un fichier Excel de test
df = pd.DataFrame({"id": [1, 2], "nom": ["Alice", "Bob"], "age": [30, 25]})
df.to_excel("test.xlsx", index=False)

# 2. Tester la conversion
with open("test.xlsx", "rb") as f:
    file = FileStorage(stream=f, filename="test.xlsx")
    result_path, error = upload_file(file)  # Appel de votre fonction

    # 3. Afficher les résultats
    if error:
        print(f"❌ ERREUR: {error}")
    else:
        print(f"✅ SUCCÈS ! Fichier converti : {result_path}")
        print("Contenu du CSV généré :")
        with open(result_path, "r") as result_file:
            print(result_file.read())

# 4. Nettoyage (supprime les fichiers temporaires)
if os.path.exists("test.xlsx"):
    os.remove("test.xlsx")
if os.path.exists(result_path):
    os.remove(result_path)  # Optionnel : supprime le CSV généré