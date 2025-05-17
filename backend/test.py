import os
import pandas as pd
from pathlib import Path
from app.core.preprocessing import (
    handle_missing_values,
    encode_data,
    normalize_data,
    select_features
)
from app.core.visualization.preprocessing import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_matrix,
    plot_missing_values
)
from app.core.visualization.results import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_true_vs_pred,
    plot_regression_errors,
    plot_clusters_2d,
    plot_silhouette,
    plot_dendrogram,
    plot_association_rules_graph
)
from app.core.algorithms import get_algorithm
from app.results.export import export_to_csv, export_to_json, export_to_excel

def choisir_fichier():
    path = Path("app/data/processed")
    fichiers = [f for f in path.iterdir() if f.suffix == ".csv"]
    print("\nFichiers disponibles :")
    for i, f in enumerate(fichiers):
        print(f"{i+1}. {f.name}")
    choix = int(input("Choisissez un fichier : ")) - 1
    return fichiers[choix]

def choisir_algo():
    algos = [
        "decision_tree", "random_forest", "knn", "svm", "naive_bayes", "neural_network",
        "linear_regression", "multiple_regression",
        "kmeans", "dbscan", "agglomerative",
        "apriori"
    ]
    print("\nAlgorithmes disponibles :")
    for i, a in enumerate(algos):
        print(f"{i+1}. {a}")
    choix = int(input("Choisissez un algorithme : ")) - 1
    return algos[choix]

def exporter(metrics, algo_name, fmt):
    if fmt == "csv":
        return export_to_csv(metrics, f"{algo_name}_results")
    elif fmt == "json":
        return export_to_json(metrics, f"{algo_name}_results")
    elif fmt == "excel":
        return export_to_excel(metrics, f"{algo_name}_results")
    return None

def main():
    # 1. Choix fichier + algo
    file_path = choisir_fichier()
    algo_name = choisir_algo()
    target_col = input("Nom de la colonne cible (si applicable) : ") if algo_name not in ["kmeans", "dbscan", "agglomerative", "apriori"] else None
    export_fmt = input("Format d'export (csv/json/excel) [csv] : ") or "csv"

    # 2. Chargement du dataset
    df = pd.read_csv(file_path)
    print(f"\n📊 {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # 3. Visualisation avant traitement
    try:
        plot_histogram(df, df.select_dtypes(include='number').columns[0])
        plot_boxplot(df, df.select_dtypes(include='number').columns[1])
        plot_correlation_matrix(df)
        plot_missing_values(df)
    except Exception as e:
        print("Erreur lors de la visualisation avant traitement :", e)

    # 4. Prétraitement
    df = handle_missing_values(df)
    df = encode_data(df)
    df = normalize_data(df)
    if target_col:
        df = select_features(df, target_col=target_col)
    processed_path = "app/data/processed/_temp_test.csv"
    df.to_csv(processed_path, index=False)

    # 5. Exécution de l’algorithme
    algo_func = get_algorithm(algo_name)
    result = algo_func(data_path=processed_path, target_column=target_col)

    # 6. Affichage des métriques
    print("\n--- ✅ Résultats ---")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")

    # 7. Option : Visualisation manuelle
    do_visu = input("\nSouhaitez-vous afficher les visualisations ? (o/n) [n] : ").strip().lower() == "o"

    if do_visu:
        print("\n--- 📈 Visualisations ---")
        try:
            if "confusion_matrix" in result:
                plot_confusion_matrix(result["confusion_matrix"], result.get("class_labels"))
            if "roc_auc" in result:
                plot_roc_curve(result["actual"], result["predictions"])
            if "actual" in result and "predictions" in result and "r2_score" in result["metrics"]:
                plot_true_vs_pred(result["actual"], result["predictions"])
                plot_regression_errors(result["actual"], result["predictions"])
            if "cluster_labels" in result:
                plot_clusters_2d(result["visualization_data"], result["cluster_labels"])
                plot_silhouette(result["visualization_data"], result["cluster_labels"])
                plot_dendrogram(result["visualization_data"])
            if "association_rules" in result:
                plot_association_rules_graph(result["association_rules"])
        except Exception as e:
            print("❌ Erreur de visualisation :", e)

    # 8. Export
    export_path = exporter(result["metrics"], algo_name, export_fmt)
    print("📁 Exporté :", export_path)

if __name__ == "__main__":
    main()
