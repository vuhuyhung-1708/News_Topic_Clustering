import pandas as pd
import pickle
import sys
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# --- CẤU HÌNH ---
N_KEYWORDS = 20  # Số từ khóa mỗi cụm

# --- THIẾT LẬP ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# 1. Các file Model Vector hóa (Giữ nguyên)
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl')
LSA_MODEL_PATH = os.path.join(PROCESSED_DIR, 'lsa_model.pkl')

# 2. ĐƯỜNG DẪN ĐẾN KẾT QUẢ DBSCAN (Vừa chạy xong)
DBSCAN_RESULT_DIR = os.path.join(PROCESSED_DIR, 'dbscan_clustered_results')
CLEAN_CSV_PATH = os.path.join(DBSCAN_RESULT_DIR, 'dbscan_clustered_results.csv')

# --- 1. NẠP DỮ LIỆU ---
print(f"--- PHÂN TÍCH CHỦ ĐỀ TỪ DBSCAN ---")
try:
    if not os.path.exists(CLEAN_CSV_PATH):
        print(f"Lỗi: Không tìm thấy file '{CLEAN_CSV_PATH}'")
        sys.exit(1)

    # Nạp Vectorizer & LSA
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    lsa_model = pickle.load(open(LSA_MODEL_PATH, 'rb'))
    
    # Nạp dữ liệu CSV
    df = pd.read_csv(CLEAN_CSV_PATH)
    
    # Lọc bỏ nhiễu (DBSCAN gán nhãn -1 là nhiễu)
    df_clean = df[df['cluster'] != -1].copy()
    df_noise = df[df['cluster'] == -1]
    
    print(f"Tổng số bài báo: {len(df)}")
    print(f"Số bài báo nhiễu (bỏ qua): {len(df_noise)}")
    print(f"Số bài báo sạch (phân tích): {len(df_clean)}")
    
    # Lấy danh sách các cụm tìm được
    unique_clusters = sorted(df_clean['cluster'].unique())
    print(f"Số lượng chủ đề tìm thấy: {len(unique_clusters)}")

except Exception as e:
    print(f"Lỗi nạp dữ liệu: {e}")
    sys.exit(1)

# --- 2. TÍNH TOÁN TÂM CỤM (Bước này dành riêng cho DBSCAN) ---
print("\nĐang tính toán tâm cụm và từ khóa...")

# Vector hóa lại dữ liệu sạch để lấy ma trận tọa độ LSA
corpus = df_clean['processed_content'].tolist()
X_tfidf = vectorizer.transform(corpus)
X_lsa = lsa_model.transform(X_tfidf)

# Chuẩn bị để giải mã từ khóa
terms = vectorizer.get_feature_names_out()
svd = lsa_model.named_steps['truncatedsvd']

# --- 3. IN KẾT QUẢ ---
# Sắp xếp các cụm theo số lượng bài báo giảm dần
cluster_counts = df_clean['cluster'].value_counts()

print("\n" + "="*70)
print("DANH SÁCH CHỦ ĐỀ (DBSCAN RESULTS)")
print("="*70 + "\n")

rank = 1
for cluster_id in cluster_counts.index:
    count = cluster_counts[cluster_id]
    
    # Lấy các vector thuộc cụm hiện tại
    # (Indices trong X_lsa khớp với thứ tự trong df_clean)
    indices_in_cluster = np.where(df_clean['cluster'] == cluster_id)[0]
    vectors_in_cluster = X_lsa[indices_in_cluster]
    
    # 1. TÍNH TÂM CỤM (Mean Center)
    # Vì DBSCAN không có tâm, ta lấy trung bình cộng các điểm để làm tâm giả định
    centroid = vectors_in_cluster.mean(axis=0).reshape(1, -1)
    
    # 2. TÌM TỪ KHÓA (Dựa trên tâm giả định)
    centroid_tfidf = svd.inverse_transform(centroid)
    sorted_term_indices = centroid_tfidf[0].argsort()[::-1]
    top_keywords = [terms[idx] for idx in sorted_term_indices[:N_KEYWORDS]]
    
    # 3. TÌM BÀI BÁO ĐẠI DIỆN (Gần tâm giả định nhất)
    # Tính khoảng cách từ các điểm trong cụm đến tâm giả định
    dists = euclidean_distances(vectors_in_cluster, centroid).flatten()
    
    # Lấy index của bài gần nhất
    closest_local_idx = dists.argmin()
    closest_global_idx = df_clean.index[indices_in_cluster[closest_local_idx]]
    
    representative_title = df_clean.loc[closest_global_idx, 'title']
    
    # In kết quả
    print(f"🔥 TOP {rank}: CHỦ ĐỀ {cluster_id} (Số lượng: {count} bài)")
    print(f"   🔑 Từ khóa: {', '.join(top_keywords)}")
    print(f"   📰 Tiêu đề đại diện: \"{representative_title}\"")
    print("-" * 70)
    
    rank += 1

print("\nHoàn thành.")