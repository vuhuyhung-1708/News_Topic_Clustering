import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MATRIX_PATH = "data/processed/lsa_matrix.pkl"
CSV_PATH = "data/processed/clustered_results/clustered_lsa_k25.csv"

# load data
with open(MATRIX_PATH, 'rb') as f:
    X = pickle.load(f)

df = pd.read_csv(CSV_PATH)

clusters = df['cluster'].unique()
clusters.sort()

print("ĐÁNH GIÁ TỪNG CỤM:")
print("-----------------------------------")

for c in clusters:
    idx = np.where(df['cluster'] == c)[0]
    points = X[idx]

    # centroid KMeans không cần load vì đã có trong clustering
    centroid = points.mean(axis=0).reshape(1, -1)

    sims = cosine_similarity(points, centroid).flatten()

    print(f"\nCụm {c}:")
    print(f"  Số bài báo: {len(idx)}")
    print(f"  Độ mạnh cụm (mean similarity): {sims.mean():.4f}")
    
    # bài báo đại diện nhất (medoid)
    best_idx = idx[np.argmax(sims)]
    title = df.iloc[best_idx]['title'] if 'title' in df.columns else "(không có tiêu đề)"
    print(f"  🎯 Bài đại diện nhất: {best_idx} — {title}")
    print(f"  Điểm similarity đại diện: {sims.max():.4f}")
