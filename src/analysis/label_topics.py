import pandas as pd
import pickle
import sys
import os
import numpy as np

# --- CẤU HÌNH ---
# Số lượng bài báo đại diện muốn in ra
N_REPRESENTATIVES = 4
# Số lượng từ khóa muốn in ra
N_KEYWORDS = 30

# --- THIẾT LẬP ĐƯỜNG DẪN (Tự động tìm theo vị trí file) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TWO_STAGE_DIR = os.path.join(PROCESSED_DIR, 'main_clustered_results')

# 1. Các file Model (Dùng để biến đổi văn bản)
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl')
LSA_MODEL_PATH = os.path.join(PROCESSED_DIR, 'lsa_model.pkl')

# 2. File Model K-Means (Của quy trình 2 giai đoạn)
KMEANS_MODEL_PATH = os.path.join(TWO_STAGE_DIR, 'kmeans_model_k30.pkl')

# 3. File Dữ liệu SẠCH (Quan trọng: Dùng file này thay vì file gốc)
CLEAN_CSV_PATH = os.path.join(TWO_STAGE_DIR, 'kmeans_clusters_k30.csv')

# --- 1. NẠP DỮ LIỆU ---
print(f"--- PHÂN TÍCH CHỦ ĐỀ NỔI BẬT (TWO-STAGE CLUSTERING) ---")
print("Đang nạp dữ liệu sạch và các model...")
try:
    # Kiểm tra file tồn tại
    if not os.path.exists(CLEAN_CSV_PATH):
        print(f"Lỗi: Không tìm thấy file '{CLEAN_CSV_PATH}'.")
        print("Hãy chạy file 'src/3_modeling/two_stage_clustering.py' trước.")
        sys.exit(1)

    # Nạp model
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    lsa_model = pickle.load(open(LSA_MODEL_PATH, 'rb'))
    kmeans = pickle.load(open(KMEANS_MODEL_PATH, 'rb'))
    
    # Nạp dữ liệu sạch
    df_clean = pd.read_csv(CLEAN_CSV_PATH)
    
    # Làm sạch lại lần nữa để đảm bảo không lỗi
    df_clean = df_clean.dropna(subset=['processed_content'])
    df_clean = df_clean[df_clean['processed_content'].astype(str).str.strip() != '']
    df_clean.reset_index(drop=True, inplace=True) # Reset index để khớp với ma trận sắp tạo

    print(f"Nạp thành công. Số lượng bài báo sạch: {len(df_clean)}")
    print(f"Số lượng cụm (K): {kmeans.n_clusters}")

except Exception as e:
    print(f"Lỗi nạp dữ liệu: {e}")
    sys.exit(1)


print("Đang vector hóa lại dữ liệu sạch để tính khoảng cách (vui lòng đợi)...")
try:
    corpus_clean = df_clean['processed_content'].tolist()
    X_tfidf = vectorizer.transform(corpus_clean)
    X_lsa_clean = lsa_model.transform(X_tfidf)
    
    # Tính khoảng cách từ các bài báo sạch đến các tâm cụm
    distances = kmeans.transform(X_lsa_clean)
except Exception as e:
    print(f"Lỗi khi tính toán vector: {e}")
    sys.exit(1)

# --- 3. CHUẨN BỊ TỪ KHÓA ---
print("Đang giải mã từ khóa...")
terms = vectorizer.get_feature_names_out()
centroids_lsa = kmeans.cluster_centers_
svd = lsa_model.named_steps['truncatedsvd']
centroids_tfidf = svd.inverse_transform(centroids_lsa)
sorted_term_indices = centroids_tfidf.argsort()[:, ::-1]

# --- 4. IN KẾT QUẢ ---
# Đếm số lượng bài báo trong mỗi cụm và sắp xếp giảm dần (để hiện chủ đề nổi bật nhất trước)
cluster_counts = df_clean['cluster'].value_counts().sort_values(ascending=False)

print("\n" + "="*60)
print("DANH SÁCH CHỦ ĐỀ THEO ĐỘ NỔI BẬT (SỐ LƯỢNG BÀI BÁO)")
print("="*60 + "\n")

rank = 1
for cluster_id, count in cluster_counts.items():
    # 1. Lấy từ khóa
    top_indices = sorted_term_indices[cluster_id, :N_KEYWORDS]
    top_keywords = [terms[idx] for idx in top_indices]
    
    # 2. Tìm bài báo đại diện
    # Lấy index của các bài báo thuộc cụm này trong df_clean
    indices_in_cluster = df_clean.index[df_clean['cluster'] == cluster_id].tolist()
    
    representative_title = "(Không tìm thấy tiêu đề hợp lệ)"
    
    if len(indices_in_cluster) > 0:
        # Lấy khoảng cách tương ứng
        cluster_dists = distances[indices_in_cluster, cluster_id]
        
        # Sắp xếp theo khoảng cách tăng dần (gần tâm nhất)
        # Lấy top 20 bài gần nhất để lọc dần
        sorted_local_indices = np.argsort(cluster_dists)[:20] 
        
        # Duyệt qua các ứng viên để tìm bài có tiêu đề tốt nhất
        for local_idx in sorted_local_indices:
            original_idx = indices_in_cluster[local_idx]
            title = str(df_clean.loc[original_idx, 'title']).strip()
            
            # Bộ lọc: Bỏ qua nếu tiêu đề là "Không tìm thấy" hoặc quá ngắn
            if title.lower() != 'không tìm thấy' and len(title) > 10:
                representative_title = title
                break

    # 3. In thông tin
    print(f"🔥 TOP {rank}: CHỦ ĐỀ {cluster_id} (Số lượng: {count} bài báo)")
    print(f"   🔑 Từ khóa: {', '.join(top_keywords)}")
    print(f"   📰 Tiêu đề đại diện: \"{representative_title}\"")
    print("-" * 60)
    
    rank += 1

print("\nHoàn thành phân tích.")