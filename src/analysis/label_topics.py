import pandas as pd
import pickle
import sys
import os
import numpy as np

# --- CẤU HÌNH ---
# Số lượng bài báo đại diện muốn xem
N_REPRESENTATIVES = 1 

# --- ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TWO_STAGE_DIR = os.path.join(PROCESSED_DIR, 'two_stage_results')

# 1. Các file Model (Dùng để biến đổi văn bản)
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl')
LSA_MODEL_PATH = os.path.join(PROCESSED_DIR, 'lsa_model.pkl')

# 2. File Model K-Means (Của quy trình 2 giai đoạn)
KMEANS_MODEL_PATH = os.path.join(TWO_STAGE_DIR, 'kmeans_two_stage.pkl')

# 3. File Dữ liệu SẠCH (Quan trọng: Dùng file này thay vì file gốc)
CLEAN_CSV_PATH = os.path.join(TWO_STAGE_DIR, 'two_stage_clusters.csv')

# --- 1. NẠP DỮ LIỆU ---
print(f"--- PHÂN TÍCH CHỦ ĐỀ NỔI BẬT (TWO-STAGE CLUSTERING) ---")
print("Đang nạp dữ liệu sạch và các model...")
try:
    # Nạp model
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    lsa_model = pickle.load(open(LSA_MODEL_PATH, 'rb'))
    kmeans = pickle.load(open(KMEANS_MODEL_PATH, 'rb'))
    
    # Nạp dữ liệu sạch
    df_clean = pd.read_csv(CLEAN_CSV_PATH)
    # Đảm bảo không có giá trị rỗng
    df_clean = df_clean.dropna(subset=['processed_content'])
    df_clean = df_clean[df_clean['processed_content'].str.strip() != '']
    df_clean.reset_index(drop=True, inplace=True)

    print(f"Nạp thành công. Số lượng bài báo sạch: {len(df_clean)}")
    print(f"Số lượng cụm (K): {kmeans.n_clusters}")

except Exception as e:
    print(f"Lỗi nạp dữ liệu: {e}")
    print("Hãy đảm bảo bạn đã chạy 'two_stage_clustering.py' trước.")
    sys.exit(1)

# --- 2. TÁI TẠO VECTOR CHO DỮ LIỆU SẠCH ---
# Bước này cực quan trọng: Chúng ta phải biến đổi dữ liệu sạch về dạng vector 
# để tính khoảng cách tới tâm cụm.
print("Đang vector hóa lại dữ liệu sạch để tính khoảng cách...")
corpus_clean = df_clean['processed_content'].tolist()
X_tfidf = vectorizer.transform(corpus_clean)
X_lsa_clean = lsa_model.transform(X_tfidf)

# Tính khoảng cách từ các bài báo sạch đến các tâm cụm
distances = kmeans.transform(X_lsa_clean)

# --- 3. CHUẨN BỊ TỪ KHÓA ---
terms = vectorizer.get_feature_names_out()
centroids_lsa = kmeans.cluster_centers_
svd = lsa_model.named_steps['truncatedsvd']
centroids_tfidf = svd.inverse_transform(centroids_lsa)
sorted_term_indices = centroids_tfidf.argsort()[:, ::-1]

# --- 4. IN KẾT QUẢ ---
# Đếm số lượng bài báo trong mỗi cụm (từ file sạch) và sắp xếp
cluster_counts = df_clean['cluster'].value_counts().sort_values(ascending=False)

print("\n--- DANH SÁCH CHỦ ĐỀ THEO ĐỘ NỔI BẬT ---\n")

rank = 1
for cluster_id, count in cluster_counts.items():
    # Lấy index của các bài báo thuộc cụm này trong df_clean
    indices_in_cluster = df_clean[df_clean['cluster'] == cluster_id].index
    
    # Lấy khoảng cách tương ứng
    cluster_distances = distances[indices_in_cluster, cluster_id]
    
    # Tìm bài gần tâm nhất
    # argsort trả về vị trí trong mảng cluster_distances
    local_min_indices = np.argsort(cluster_distances)[:10] # Lấy top 10 để lọc
    
    # Map về index của df_clean
    candidate_indices = indices_in_cluster[local_min_indices]
    
    # Chọn bài báo đại diện (Lọc tiêu đề ngắn/lỗi)
    representative_title = "(Không tìm thấy tiêu đề hợp lệ)"
    for idx in candidate_indices:
        title = str(df_clean.loc[idx, 'title']).strip()
        if title.lower() != "không tìm thấy" and len(title) > 10:
            representative_title = title
            break
            
    # Lấy từ khóa
    top_indices = sorted_term_indices[cluster_id, :10]
    top_keywords = [terms[idx] for idx in top_indices]

    print(f"   TOP {rank}: CHỦ ĐỀ {cluster_id} (Số lượng: {count} bài báo)")
    print(f"   Từ khóa: {', '.join(top_keywords)}")
    print(f"   Tiêu đề đại diện: \"{representative_title}\"")
    
    print("-" * 60)
    rank += 1