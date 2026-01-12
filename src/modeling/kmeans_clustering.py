# src/modeling/kmeans_clustering.py

"""
K-Means clustering on LSA-reduced TF-IDF vectors
This is the MAIN clustering pipeline of the system
"""

import os
import pickle
import pandas as pd
from sklearn.cluster import KMeans

# ================= CONFIG =================
K_VALUE = 26

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

LSA_MATRIX_PATH = os.path.join(PROCESSED_DIR, "lsa_matrix.pkl")
DATA_PATH = os.path.join(PROCESSED_DIR, "processed_data.csv")

OUTPUT_DIR = os.path.join(PROCESSED_DIR, "kmeans_clustered_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD DATA =================
print("📥 Loading LSA matrix and processed data...")

with open(LSA_MATRIX_PATH, "rb") as f:
    X = pickle.load(f)

df = pd.read_csv(DATA_PATH)
df.dropna(subset=["processed_content"], inplace=True)
df = df[df["processed_content"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)

print(f"✅ Loaded {len(df)} documents")

# ================= K-MEANS =================
print(f"🚀 Running K-Means clustering (K={K_VALUE})...")

kmeans = KMeans(
    n_clusters=K_VALUE,
    random_state=RANDOM_STATE,
    n_init="auto"
)

labels = kmeans.fit_predict(X)
df["cluster"] = labels

# ================= SAVE RESULTS =================
cluster_csv = os.path.join(
    OUTPUT_DIR, f"kmeans_clusters_k{K_VALUE}.csv"
)
model_path = os.path.join(
    OUTPUT_DIR, f"kmeans_model_k{K_VALUE}.pkl"
)

df.to_csv(cluster_csv, index=False, encoding="utf-8-sig")

with open(model_path, "wb") as f:
    pickle.dump(kmeans, f)

print("✅ K-Means clustering completed")
print(f"📁 Results saved to: {OUTPUT_DIR}")
print(df["cluster"].value_counts().sort_index())
