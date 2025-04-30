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