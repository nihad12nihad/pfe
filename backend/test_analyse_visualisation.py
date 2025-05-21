import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import arff

from app.core.preprocessing import (
    handle_missing_values,
    encode_data,
    normalize_data,
    select_features
)
from app.core.algorithms import get_algorithm
from app.results.export import export_to_csv, export_to_json, export_to_excel

from app.core.visualization.charts import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_matrix,
    plot_missing_values,
    plot_unique_values
)

from app.core.visualization.results import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_regression_results,
    plot_regression_errors,
    plot_clusters_2d,
    plot_silhouette_scores,
    plot_association_rules_heatmap
)

def save_and_show(fig, name):
    output_dir = "app/data/resultats"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Image enregistrée : {path}")

def load_file(filepath):
    ext = filepath.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext == ".xlsx":
        return pd.read_excel(filepath)
    elif ext == ".json":
        return pd.read_json(filepath)
    elif ext == ".txt":
        return pd.read_csv(filepath, delimiter="\t")
    elif ext == ".arff":
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        for col in df.select_dtypes([object]):
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        return df
    else:
        raise ValueError(f"Format non supporté : {ext}")

def choisir_fichier():
    path = Path("app/data/raw")
    fichiers = [f for f in path.iterdir() if f.suffix in [".csv", ".xlsx", ".json", ".txt", ".arff"]]
    print("\n📂 Fichiers disponibles :")
    for i, f in enumerate(fichiers):
        print(f"{i+1}. {f.name}")
    choix = int(input("👉 Choisissez un fichier (n°) : ")) - 1
    return fichiers[choix]

def choisir_algo():
    algos = [
        "decision_tree", "random_forest", "knn", "svm", "naive_bayes", "neural_network",
        "linear_regression", "multiple_regression",
        "kmeans", "dbscan", "agglomerative",
        "apriori"
    ]
    print("\n⚙️ Algorithmes disponibles :")
    for i, a in enumerate(algos):
        print(f"{i+1}. {a}")
    choix = int(input("👉 Choisissez un algorithme (n°) : ")) - 1
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
    file_path = choisir_fichier()

    # Chargement du fichier
    df = load_file(file_path)
    print(f"\n✅ Fichier chargé : {file_path.name}")
    print(f"📏 {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Si fichier non-CSV, sauvegarder en CSV dans app/data/raw
    if file_path.suffix.lower() != ".csv":
        raw_dir = Path("app/data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        converted_path = raw_dir / f"{file_path.stem}.csv"
        df.to_csv(converted_path, index=False)
        file_path = converted_path
        print(f"📄 Fichier converti et sauvegardé sous : {converted_path}")

    algo_name = choisir_algo()
    need_target = algo_name not in ["kmeans", "dbscan", "agglomerative", "apriori"]
    target_col = input("🎯 Nom de la colonne cible (si applicable) : ") if need_target else None
    export_fmt = input("💾 Format d’export (csv/json/excel) [csv] : ") or "csv"

    # Afficher les colonnes disponibles pour vérifier la cible
    print(f"🧾 Colonnes du fichier : {list(df.columns)}")
    if target_col and target_col not in df.columns:
        print(f"❌ La colonne cible '{target_col}' est introuvable.")
        return

    # 🔹 Visualisation avant traitement
    print("\n📊 --- Visualisation avant prétraitement ---")
    try:
        for col in df.select_dtypes(include="number").columns[:2]:
            save_and_show(plot_histogram(df, col), f"histogram_{col}")
            save_and_show(plot_boxplot(df, col), f"boxplot_{col}")
        save_and_show(plot_correlation_matrix(df), "correlation_matrix")
        mv_fig = plot_missing_values(df)
        if mv_fig:
            save_and_show(mv_fig, "missing_values")
        save_and_show(plot_unique_values(df), "unique_values")
    except Exception as e:
        print("❌ Erreur de visualisation avant analyse :", e)

    # 🔹 Prétraitement
    print("\n⚙️ Prétraitement en cours...")
    if target_col:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
        X = handle_missing_values(X)
        X = encode_data(X)
        X = normalize_data(X)
        df = pd.concat([X, y], axis=1)
        cols = [col for col in df.columns if col != target_col] + [target_col]
        df = df[cols]
        df = select_features(df, target_col=target_col)
    else:
        df = handle_missing_values(df)
        df = encode_data(df)
        df = normalize_data(df)

    processed_path = "app/data/processed/_temp_test.csv"
    df.to_csv(processed_path, index=False)

    # 🔹 Analyse
    print("\n🚀 --- Exécution de l'algorithme ---")
    algo_func = get_algorithm(algo_name)
    result = algo_func(data_path=processed_path, target_column=target_col)

    # 🔹 Résultats
    print("\n📈 --- Résultats ---")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")

    # 🔹 Visualisation après analyse
    print("\n🖼️ --- Visualisation après analyse ---")
    try:
        if "confusion_matrix" in result:
            fig = plot_confusion_matrix(result["actual"], result["predictions"])
            save_and_show(fig, "confusion_matrix")

        if "roc_auc" in result and "probabilities" in result:
            fig = plot_roc_curve(result["actual"], result["probabilities"])
            save_and_show(fig, "roc_curve")
            fig = plot_precision_recall_curve(result["actual"], result["probabilities"])
            save_and_show(fig, "precision_recall")

        if "r2_score" in result["metrics"] and "predictions" in result:
            fig = plot_regression_results(result["actual"], result["predictions"])
            save_and_show(fig, "regression_plot")
            fig = plot_regression_errors(result["actual"], result["predictions"])
            save_and_show(fig, "regression_errors")

        if "cluster_labels" in result and "visualization_data" in result:
            fig = plot_clusters_2d(result["visualization_data"], result["cluster_labels"])
            save_and_show(fig, "clusters_2d")
            fig = plot_silhouette_scores(result["visualization_data"], result["cluster_labels"])
            save_and_show(fig, "silhouette")

        if "association_rules" in result:
            fig = plot_association_rules_heatmap(result["association_rules"])
            save_and_show(fig, "association_rules")

    except Exception as e:
        print("❌ Erreur visualisation après analyse :", e)

    # 🔹 Export
    export_path = exporter(result["metrics"], algo_name, export_fmt)
    print("📤 Export effectué :", export_path)

if __name__ == "__main__":
    main()
