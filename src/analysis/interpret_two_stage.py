import pandas as pd
import pickle
import sys
import os
import numpy as np

# --- CẤU HÌNH ---
K_VALUE = 34
N_KEYWORDS = 15
N_REPRESENTATIVES = 3
N_CANDIDATES = 20 # Lấy nhiều ứng viên để lọc bài lỗi

PROCESSED_DIR = 'data/processed'
CSV_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl')
LSA_MODEL_PATH = os.path.join(PROCESSED_DIR, 'lsa_model.pkl')
MODEL_DIR = os.path.join(PROCESSED_DIR, 'two_stage_results')
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, f'kmeans_two_stage.pkl')
MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl')

print(f"--- PHÂN TÍCH CHỦ ĐỀ (K={K_VALUE}) ---")
try:
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    lsa_model = pickle.load(open(LSA_MODEL_PATH, 'rb'))
    kmeans = pickle.load(open(KMEANS_MODEL_PATH, 'rb'))
    with open(MATRIX_PATH, 'rb') as f:
        X_lsa = pickle.load(f)
    
    df = pd.read_csv(CSV_PATH)
    df.dropna(subset=['processed_content'], inplace=True)
    df = df[df['processed_content'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)
    print("Nạp dữ liệu thành công.")
except Exception as e:
    print(f"Lỗi nạp dữ liệu: {e}")
    sys.exit(1)


# Gán nhãn cụm cho từng bài báo
df['cluster'] = kmeans.predict(X_lsa)
# Tính khoảng cách từ bài báo đến các tâm cụm
distances = kmeans.transform(X_lsa)

# Giải mã từ khóa (LSA -> TF-IDF)
terms = vectorizer.get_feature_names_out()
centroids_lsa = kmeans.cluster_centers_
svd = lsa_model.named_steps['truncatedsvd']
centroids_tfidf = svd.inverse_transform(centroids_lsa)
sorted_term_indices = centroids_tfidf.argsort()[:, ::-1]


print("\n--- KẾT QUẢ GÁN NHÃN ---")

for i in range(K_VALUE):
    print(f"\n==================== CHỦ ĐỀ {i} ====================")
    
    # A. In từ khóa
    top_indices = sorted_term_indices[i, :N_KEYWORDS]
    top_keywords = [terms[idx] for idx in top_indices]
    print(f"  TỪ KHÓA: {', '.join(top_keywords)}")
    
    # B. Tìm bài báo đại diện
    indices_in_cluster = df[df['cluster'] == i].index
    if len(indices_in_cluster) == 0:
        continue
        
    # Lấy khoảng cách của các bài trong cụm đến tâm cụm
    cluster_distances = distances[indices_in_cluster, i]
    # Lấy top các bài gần tâm nhất
    local_indices = np.argsort(cluster_distances)[:N_CANDIDATES]
    candidate_indices = indices_in_cluster[local_indices]
    
    print("  BÀI BÁO ĐẠI DIỆN:")
    count_found = 0
    for idx in candidate_indices:
        title = str(df.loc[idx, 'title']).strip()
        
        # Lọc bỏ bài báo lỗi hoặc tiêu đề rỗng
        if title.lower() == "không tìm thấy" or len(title) < 10:
            continue
            
        print(f"    #{count_found + 1}: {title}")
        count_found += 1
        if count_found >= N_REPRESENTATIVES:
            break