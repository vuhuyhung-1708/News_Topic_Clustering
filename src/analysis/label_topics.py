import pandas as pd
import pickle
import os
import numpy as np

# --- CẤU HÌNH ---
N_REPRESENTATIVES = 4
N_KEYWORDS = 30


# --- ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TWO_STAGE_DIR = os.path.join(PROCESSED_DIR, 'kmeans_clustered_results')

VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl')
LSA_MODEL_PATH = os.path.join(PROCESSED_DIR, 'lsa_model.pkl')
KMEANS_MODEL_PATH = os.path.join(TWO_STAGE_DIR, 'kmeans_model_k28.pkl')
CLEAN_CSV_PATH = os.path.join(TWO_STAGE_DIR, 'kmeans_clusters_k28.csv')


# --- 1. NẠP MODEL & DỮ LIỆU ---
vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
lsa_model = pickle.load(open(LSA_MODEL_PATH, 'rb'))
kmeans = pickle.load(open(KMEANS_MODEL_PATH, 'rb'))

df = pd.read_csv(CLEAN_CSV_PATH)
df = df.dropna(subset=['processed_content'])
df = df[df['processed_content'].astype(str).str.strip() != '']
df.reset_index(drop=True, inplace=True)

print(f"✔ Số bài báo: {len(df)}")
print(f"✔ Số cụm (K): {kmeans.n_clusters}")

# --- 2. VECTOR HÓA ---
corpus = df['processed_content'].tolist()
X_tfidf = vectorizer.transform(corpus)
X_lsa = lsa_model.transform(X_tfidf)

# Khoảng cách tới tâm cụm
distances = kmeans.transform(X_lsa)

# --- 3. GIẢI MÃ TỪ KHÓA ---
terms = vectorizer.get_feature_names_out()
centroids_lsa = kmeans.cluster_centers_

svd = lsa_model.named_steps['truncatedsvd']
centroids_tfidf = svd.inverse_transform(centroids_lsa)
sorted_terms = centroids_tfidf.argsort()[:, ::-1]

# --- 4. IN KẾT QUẢ ---
cluster_sizes = df['cluster'].value_counts().sort_values(ascending=False)

print("\n" + "=" * 60)
print("DANH SÁCH CHỦ ĐỀ THEO ĐỘ NỔI BẬT")
print("=" * 60 + "\n")

rank = 1
for cluster_id, count in cluster_sizes.items():
    # Từ khóa
    top_keywords = [
        terms[i] for i in sorted_terms[cluster_id, :N_KEYWORDS]
    ]

    # Bài báo đại diện 
    cluster_indices = df.index[df['cluster'] == cluster_id]
    cluster_distances = distances[cluster_indices, cluster_id]

    representative_title = "(Không tìm thấy tiêu đề phù hợp)"
    for idx in cluster_indices[np.argsort(cluster_distances)[:20]]:
        title = str(df.loc[idx, 'title']).strip()
        if title.lower() != 'không tìm thấy' and len(title) > 10:
            representative_title = title
            break

    print(f"🔥 TOP {rank}: CHỦ ĐỀ {cluster_id} ({count} bài báo)")
    print(f"   🔑 Từ khóa: {', '.join(top_keywords)}")
    print(f"   📰 Bài báo đại diện: \"{representative_title}\"")
    print("-" * 60)

    rank += 1

print("\n✔ Hoàn thành phân tích.")
