from sklearn import datasets
import pandas as pd
import os

# Créer le dossier s'il n'existe pas
os.makedirs('datasets', exist_ok=True)

# === Classification ===
# Iris
iris = datasets.load_iris(as_frame=True)
iris_df = iris.frame
iris_df.to_csv('datasets/iris.csv', index=False)

# Wine
wine = datasets.load_wine(as_frame=True)
wine_df = wine.frame
wine_df.to_csv('datasets/wine.csv', index=False)

# Breast Cancer
cancer = datasets.load_breast_cancer(as_frame=True)
cancer_df = cancer.frame
cancer_df.to_csv('datasets/breast_cancer.csv', index=False)


# === Régression ===
# Boston alternative : California Housing
california = datasets.fetch_california_housing(as_frame=True)
california_df = california.frame
california_df.to_csv('datasets/california_housing.csv', index=False)


# === Clustering ===
# Blobs
blobs_data, blobs_labels = datasets.make_blobs(n_samples=300, centers=3, random_state=42)
blobs_df = pd.DataFrame(blobs_data, columns=['Feature1', 'Feature2'])
blobs_df['Cluster'] = blobs_labels
blobs_df.to_csv('datasets/blobs.csv', index=False)

# Moons
moons_data, moons_labels = datasets.make_moons(n_samples=300, noise=0.1, random_state=42)
moons_df = pd.DataFrame(moons_data, columns=['Feature1', 'Feature2'])
moons_df['Label'] = moons_labels
moons_df.to_csv('datasets/moons.csv', index=False)


# === Association ===
# Market Basket Simulation
transactions = [
    ["milk", "bread", "eggs"],
    ["bread", "diapers", "beer", "eggs"],
    ["milk", "bread", "diapers", "beer"],
    ["bread", "diapers", "beer"],
    ["milk", "bread", "diapers", "eggs"]
]
market_basket_df = pd.DataFrame({'Transaction': [";".join(items) for items in transactions]})
market_basket_df.to_csv('datasets/market_basket.csv', index=False)

print("✅ Tous les datasets sont générés dans le dossier 'datasets/'")
