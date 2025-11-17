import pandas as pd
import pickle
import sys
import os
import numpy as np

# --- CẤU HÌNH ---
# Tên file kết quả DBSCAN bạn muốn phân tích
DBSCAN_RESULT_FILE = 'dbscan_eps0.5_min5.csv'

# --- ĐƯỜNG DẪN ĐƠN GIẢN HÓA ---
PROCESSED_DIR = 'data/processed'
# Ma trận LSA "tốt nhất" (vector 300 chiều)
MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl') 
# "Từ điển" TF-IDF gốc (5000 từ)
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl') 
# "Bộ nén" LSA
LSA_MODEL_PATH = os.path.join(PROCESSED_DIR, 'lsa_model.pkl') 
# File kết quả DBSCAN
DBSCAN_CSV_PATH = os.path.join(PROCESSED_DIR, 'clustered_results_dbscan', DBSCAN_RESULT_FILE)

# --- 1. NẠP DỮ LIỆU VÀ MODEL ---
print(f"--- Đang phân tích kết quả cho file {DBSCAN_RESULT_FILE} ---")
print("Nạp ma trận LSA, file CSV kết quả, và các model...")
try:
    # Nạp ma trận LSA (X)
    with open(MATRIX_PATH, 'rb') as f:
        X_lsa = pickle.load(f)
    # Nạp "Từ điển" TF-IDF
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    # Nạp "Bộ nén" LSA
    lsa_model = pickle.load(open(LSA_MODEL_PATH, 'rb'))
    # Nạp file CSV kết quả DBSCAN
    df = pd.read_csv(DBSCAN_CSV_PATH)
    
    # Lấy ra các nhãn cụm duy nhất (loại bỏ -1 là nhiễu)
    unique_labels = sorted([label for label in df['cluster'].unique() if label != -1])
    n_clusters = len(unique_labels)
    
    print(f"Nạp dữ liệu thành công. Phát hiện {n_clusters} cụm (bỏ qua nhiễu).")

except FileNotFoundError as e:
    print("\nLỗi: Không tìm thấy file model hoặc file kết quả DBSCAN.")
    print(f"Chi tiết lỗi: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi khi nạp file: {e}")
    sys.exit(1)

# --- 2. TÍNH TOÁN TÂM CỤM "GIẢ LẬP" CHO DBSCAN ---
print("\nĐang tính toán tâm cụm (trung bình cộng) cho các cụm DBSCAN...")
centroids_lsa = []
for cluster_id in unique_labels:
    # Lấy index (vị trí) của tất cả bài báo thuộc cụm này
    indices = df[df['cluster'] == cluster_id].index
    
    # Lấy ra các vector LSA của các bài báo đó
    cluster_vectors = X_lsa[indices]
    
    # Tính vector trung bình cộng (đây chính là tâm cụm "giả lập")
    centroid_lsa = np.mean(cluster_vectors, axis=0)
    centroids_lsa.append(centroid_lsa)

# Chuyển danh sách các tâm cụm thành một ma trận numpy
centroids_lsa_matrix = np.array(centroids_lsa)

# --- 3. TRÍCH XUẤT TỪ KHÓA (GIỐNG HỆT K-MEANS) ---
print("Trích xuất từ khóa cho từng chủ đề (DBSCAN)...")
try:
    terms = vectorizer.get_feature_names_out()
    svd = lsa_model.named_steps['truncatedsvd']
    centroids_tfidf = svd.inverse_transform(centroids_lsa_matrix)
    sorted_term_indices = centroids_tfidf.argsort()[:, ::-1]

    print(f"\n--- CÁC CHỦ ĐỀ ĐƯỢC PHÁT HIỆN (DBSCAN) ---")
    num_keywords = 15
    
    for i, cluster_id in enumerate(unique_labels):
        top_keywords_indices = sorted_term_indices[i, :num_keywords]
        top_keywords = [terms[idx] for idx in top_keywords_indices]
        # Lấy số lượng bài báo trong cụm
        count = len(df[df['cluster'] == cluster_id])
        print(f"Chủ đề {cluster_id} (Có {count} bài báo): {', '.join(top_keywords)}")
        
except Exception as e:
    print(f"Lỗi khi trích xuất từ khóa: {e}")