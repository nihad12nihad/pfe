import os
import pandas as pd
import matplotlib.pyplot as plt

from app.core.visualization.results import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)

def save_and_show(fig, name):
    output_dir = "app/data/resultats"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Image enregistrée : {path}")

def main():
    # Charge ton CSV de test qui contient au minimum 'y_true' et 'y_scores'
    csv_path = "app/data/raw/diabetes.csv"
    df = pd.read_csv(csv_path)
    
    # Extraction des vraies classes et des scores
    y_true = df["y_true"]
    y_scores = df["y_scores"]
    
    # Pour confusion matrix, on crée des prédictions binaires par seuil 0.5 (par exemple)
    y_pred = (y_scores >= 0.5).astype(int)
    
    # Matrice de confusion
    fig_cm = plot_confusion_matrix(y_true, y_pred, labels=[0,1])
    save_and_show(fig_cm, "confusion_matrix_test")
    
    # Courbe ROC
    fig_roc = plot_roc_curve(y_true, y_scores, pos_label=1)
    save_and_show(fig_roc, "roc_curve_test")
    
    # Courbe précision-rappel
    fig_pr = plot_precision_recall_curve(y_true, y_scores)
    save_and_show(fig_pr, "precision_recall_test")

if __name__ == "__main__":
    main()
