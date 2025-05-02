import os
import pandas as pd
from werkzeug.utils import secure_filename

def upload_file(file):
    # 1. Sécuriser et récupérer le nom du fichier
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    # 2. Vérifier les extensions supportées
    supported_ext = ['.csv', '.xlsx', '.txt', '.json']
    if ext not in supported_ext:
        return None, f"Erreur : Format '{ext}' non supporté. Formats acceptés : CSV, XLSX, TXT, JSON."

    # 3. Sauvegarder le fichier dans le répertoire 'data/raw/'
    upload_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')  # Chemin relatif vers 'data/raw'
    os.makedirs(upload_dir, exist_ok=True)  # Crée le dossier si il n'existe pas
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)

    try:
        # 4. Traitement selon le type de fichier
        if ext == '.csv':
            # Si le fichier est déjà CSV, on le retourne sans modification
            return filepath, None

        elif ext == '.xlsx':
            df = pd.read_excel(filepath)

        elif ext == '.txt':
            try:
                df = pd.read_csv(filepath, sep='\t')  # d'abord tabulation
            except:
                df = pd.read_csv(filepath, sep=',')  # sinon virgule

        elif ext == '.json':
            df = pd.read_json(filepath)

        # 5. Conversion en CSV
        csv_path = filepath.rsplit('.', 1)[0] + ".csv"  # Créer un chemin avec l'extension .csv
        df.to_csv(csv_path, index=False)  # Sauvegarde en CSV

        # Ne pas supprimer le fichier original, il est conservé dans 'data/raw/'
        return csv_path, None  # Retourner le chemin du fichier CSV

    except Exception as e:
        return None, f"Erreur lors de la conversion : {str(e)}"
