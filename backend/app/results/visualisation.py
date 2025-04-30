<<<<<<< HEAD
import matplotlib 
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any,Optional
import numpy as np
from pathlib import Path

def ensure_directory(path : str) -> Path :
     path=Path(path).absolute()
     path.parent.mkdir(parents=True, exist_ok=True)
     return path

def plot_classification_results(metrics: dict[str, float], algorithm_name: str, output_dir: str="backend/data/processed") -> Optional[str]:
    try:
        plt.figure(figsize=(10,5))
        ax=sns.barplot(x=list(metrics.keys()),y=list(metrics.values()))
        ax.bar_label(ax.containers[0], fmt='%.2f')
        plt.title(f"performance de {algorithm_name}",pad=20)
        plt.ylabel("score",fontweight='bold')
        plt.ylim(0,1)
        plt.grid(axis='y',alpha=0.3)
        
        plot_path=ensure_directory(f"{output_dir}/plot_{algorithm_name}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=120)
        plt.show()
        plt.close()
        return str(plot_path)
    except Exception as e:
        print(f"erreur lors de la génération du grapique : {e}")
        return None
    
def plot_confusion_matrix(cm:np.ndarray, labels: list[str],output_dir: str="backend/data/processed") -> Optional[str]:
    try:
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d",xticklabels=labels, yticklabels=labels, cmap="Blues", cbar=False)
        plt.title("matrice de confusion", pad=15)
        plt.xlabel("prédit", fontweight='bold')
        plt.ylabel("Réel", fontweight='bold')
        
        plot_path= ensure_directory(f"{output_dir}/confusion_matrix.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=120)
        plt.show()
        plt.close()
        return str(plot_path)
    except Exception as e:
        print(f"erreur lors de la génération de la matrice : {e}")
        return None
    
def plot_clusters(data: np.ndarray, labels: np.ndarray, output_dir: str="backend/data/processed", title: str="visualisation des clusters") -> Optional[str]:
    try:
        if data.shape[1]<2:
            raise ValueError("les données doivent avoir au moins 2 dismensions")
        plt.figure(figsize=(8,6))
        scatter=plt.scatter(data[:,0],data[:,1], c=labels, cmap="viridis", alpha=0.6, edgecolors='w', linewidths=0.5)
        plt.legend(*scatter.legend_elements(), title="clusters")
        plt.title(title, pad=15)
        plt.grid(alpha=0.3)
        
        plot_path=ensure_directory(f"{output_dir}/clusters.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=120)
        plt.show()
        plt.close()
        return str(plot_path)
    except Exception as e:
        print(f"errue lors de la visualisation des clusters : {e}")
        return None     



    

# --- Données de test ---
# 1. Métriques de classification
test_metrics = {
    "accuracy": 0.92,
    "precision": 0.85,
    "recall": 0.88,
    "f1": 0.86
}

# 2. Matrice de confusion (exemple binaire)
test_cm = np.array([
    [50, 5],  # Vrais négatifs | Faux positifs
    [3, 42]   # Faux négatifs  | Vrais positifs
])
test_labels = ["Négatif", "Positif"]

# 3. Données de clusters (aléatoires)
np.random.seed(42)
test_data = np.random.randn(100, 2)
test_clusters = np.random.randint(0, 3, 100)

# --- Exécution des tests ---
if __name__ == "__main__":
    # Création du dossier de sortie
    Path("backend/data/processed").mkdir(parents=True, exist_ok=True)
    
    # 1. Test classification
    print("Génération du graphique de classification...")
    classification_path = plot_classification_results(test_metrics, "RandomForest")
    
    # 2. Test matrice de confusion
    print("\nGénération de la matrice de confusion...")
    confusion_path = plot_confusion_matrix(test_cm, test_labels)
    
    # 3. Test clusters
    print("\nGénération de la visualisation des clusters...")
    clusters_path = plot_clusters(test_data, test_clusters)
    
    # Affichage des résultats
    print("\nRésultats :")
    print(f"- Classification : {classification_path}")
    print(f"- Matrice de confusion : {confusion_path}")
    print(f"- Clusters : {clusters_path}")
    
    print("\nOuvrez les fichiers PNG dans backend/data/processed/ pour voir les graphiques")
=======
import matplotlib
matplotlib.use('Agg')  # Utilise un mode sans fenêtre graphique

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Matrice de corrélation (Heatmap)
def plot_correlation_matrix(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Matrice de Corrélation')
    plt.savefig('graphique1.png')

# 2. Histogramme de distribution
def plot_histogram(df: pd.DataFrame, column: str):
    plt.figure(figsize=(8,6))
    sns.histplot(df[column].dropna(), bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution de {column}')
    plt.xlabel(column)
    plt.ylabel('Fréquence')
    plt.savefig('graphique2.png')

# 3. Boxplot pour chaque variable numérique
def plot_boxplot(df: pd.DataFrame, column: str):
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[column], color='lightcoral')
    plt.title(f'Boxplot de {column}')
    plt.xlabel(column)
    plt.savefig('graphique3.png')

# 4. Countplot pour variables catégorielles
def plot_categorical_count(df: pd.DataFrame, column: str):
    plt.figure(figsize=(8,6))
    sns.countplot(x=column, data=df, palette='muted')
    plt.title(f'Distribution de {column}')
    plt.xlabel(column)
    plt.ylabel('Nombre')
    plt.xticks(rotation=45)
    plt.savefig('graphique4.png')

# 5. Nuage de points avec droite de régression
def plot_linear_relationship(df: pd.DataFrame, x_col: str, y_col: str):
    plt.figure(figsize=(8,6))
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.title(f'Relation linéaire entre {x_col} et {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.savefig('graphique5.png')

# 6. Histogramme des valeurs manquantes
def plot_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("✅ Pas de valeurs manquantes détectées.")
        return
    plt.figure(figsize=(10,6))
    sns.barplot(x=missing.values, y=missing.index, palette='viridis')
    plt.title('Valeurs manquantes par colonne')
    plt.xlabel('Nombre de valeurs manquantes')
    plt.ylabel('Colonnes')
    plt.savefig('graphique6.png')

# 7. Nombre de valeurs uniques par colonne
def plot_unique_values(df: pd.DataFrame):
    unique_counts = df.nunique().sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=unique_counts.values, y=unique_counts.index, palette='crest')
    plt.title('Nombre de valeurs uniques par colonne')
    plt.xlabel('Nombre de valeurs uniques')
    plt.ylabel('Colonnes')
    plt.savefig('graphique7.png')




>>>>>>> bb823268afdef242edd8a69ce8173f18286257fe
