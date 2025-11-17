import pandas as pd
import pickle
import sys
import os
import numpy as np

# --- CẤU HÌNH ---
# Đặt K_VALUE khớp với model K-Means LSA tốt nhất của bạn
K_VALUE = 34 
N_KEYWORDS = 15       # Số lượng từ khóa muốn xem
N_REPRESENTATIVES = 3 # Số lượng bài báo đại diện muốn xem

# --- ĐƯỜNG DẪN ĐƠN GIẢN HÓA ---
PROCESSED_DIR = 'data/processed'
CSV_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl') # "Từ điển" TF-IDF
LSA_MODEL_PATH = os.path.join(PROCESSED_DIR, 'lsa_model.pkl') # "Bộ nén" LSA
MODEL_DIR = os.path.join(PROCESSED_DIR, 'clustered_results')
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, f'kmeans_lsa_k{K_VALUE}.pkl')

# --- 1. NẠP DỮ LIỆU VÀ MODEL ---
print(f"--- Bắt đầu gán nhãn cho K={K_VALUE} ---")
print("Nạp các file CSV và Model...")
try:
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    lsa_model = pickle.load(open(LSA_MODEL_PATH, 'rb'))
    kmeans = pickle.load(open(KMEANS_MODEL_PATH, 'rb'))
    
    df = pd.read_csv(CSV_PATH)
    df.dropna(subset=['processed_content'], inplace=True)
    df = df[df['processed_content'].str.strip() != '']
    df.reset_index(drop=True, inplace=True) # Rất quan trọng để index khớp

    print("Nạp dữ liệu thành công.")

except Exception as e:
    print(f"Lỗi khi nạp dữ liệu: {e}.")
    sys.exit(1)

# --- 2. GÁN NHÃN CỤM VÀ TÍNH KHOẢNG CÁCH ---
print("Đang gán nhãn cụm và tính khoảng cách...")
# Tải ma trận LSA "tốt nhất"
MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl')
with open(MATRIX_PATH, 'rb') as f:
    X_lsa = pickle.load(f)

if X_lsa.shape[0] != len(df):
    print(f"Lỗi: Số dòng ma trận LSA ({X_lsa.shape[0]}) và CSV ({len(df)}) không khớp.")
    sys.exit(1)

# Gán nhãn cluster vào df
df['cluster'] = kmeans.predict(X_lsa)
# Tính khoảng cách từ mỗi điểm đến TẤT CẢ các tâm cụm
distances = kmeans.transform(X_lsa)

# --- 3. TRÍCH XUẤT TỪ KHÓA VÀ BÀI BÁO ĐẠI DIỆN ---
print("\n--- KẾT QUẢ GÁN NHÃN CHỦ ĐỀ ---")

try:
    # Logic lấy từ khóa
    terms = vectorizer.get_feature_names_out()
    centroids_lsa = kmeans.cluster_centers_
    svd = lsa_model.named_steps['truncatedsvd']
    centroids_tfidf = svd.inverse_transform(centroids_lsa)
    sorted_term_indices = centroids_tfidf.argsort()[:, ::-1]

    for i in range(K_VALUE): # Lặp qua từng chủ đề
        print(f"\n==================== CHỦ ĐỀ {i} ====================")
        
        # 1. LẤY TỪ KHÓA HÀNG ĐẦU
        top_indices = sorted_term_indices[i, :N_KEYWORDS]
        top_keywords = [terms[idx] for idx in top_indices]
        print(f"  TỪ KHÓA CHÍNH: {', '.join(top_keywords)}")
        
        # 2. LẤY BÀI BÁO ĐẠI DIỆN (Logic đã sửa lỗi)
        
        # Lấy index (vị trí) của tất cả bài báo thuộc cụm i
        indices_in_cluster = df[df['cluster'] == i].index
        
        if len(indices_in_cluster) == 0:
            print("  BÀI BÁO ĐẠI DIỆN: (Không tìm thấy bài báo nào trong cụm này)")
            continue
            
        # Lấy khoảng cách của các bài báo này ĐẾN TÂM CỤM CỦA CHÍNH NÓ (tâm i)
        cluster_distances = distances[indices_in_cluster, i]
        
        # Sắp xếp các khoảng cách này (từ nhỏ đến lớn) và lấy N chỉ số đầu tiên
        # argsort() trả về chỉ số (vị trí) của các bài báo TRONG DANH SÁCH cluster_distances
        local_indices = np.argsort(cluster_distances)[:N_REPRESENTATIVES]
        
        # Chuyển đổi chỉ số cục bộ (local) về chỉ số gốc (global) của DataFrame
        representative_indices = indices_in_cluster[local_indices]
        
        print("\n  BÀI BÁO ĐẠI DIỆN:")
        for j, index in enumerate(representative_indices):
            print(f"    #{j+1}: {df.loc[index, 'title']}")

except Exception as e:
    print(f"Lỗi khi trích xuất: {e}")