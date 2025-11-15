import pickle
import sys
import os

# --- CẤU HÌNH ---
# Đặt K_VALUE khớp với model bạn muốn xem (ví dụ: 34 hoặc 35)
K_VALUE = 34 
N_COMPONENTS = 300 # Phải khớp với file feature_extraction.py

# --- ĐƯỜNG DẪN FILE ---
PROCESSED_DIR = 'data/processed'
ADVANCED_DIR = os.path.join(PROCESSED_DIR, 'clustered_results') # Thư mục chứa model K-Means LSA

# 1. "Từ điển" TF-IDF gốc (5000 từ)
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl') 
# 2. "Bộ nén" LSA (đã lưu ở bước feature_extraction)
LSA_MODEL_PATH = os.path.join(PROCESSED_DIR, 'lsa_model.pkl') 
# 3. "Bộ não" K-Means (đã học trên dữ liệu nén)
KMEANS_MODEL_PATH = os.path.join(ADVANCED_DIR, f'kmeans_lsa_k{K_VALUE}.pkl') 

# --- 1. NẠP CẢ 3 MODEL ---
print(f"--- Đang phân tích kết quả cho mô hình LSA K={K_VALUE} ---")
print(f"Nạp từ điển từ: {VECTORIZER_PATH}")
print(f"Nạp bộ nén LSA từ: {LSA_MODEL_PATH}")
print(f"Nạp model K-Means từ: {KMEANS_MODEL_PATH}")

try:
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    lsa_model = pickle.load(open(LSA_MODEL_PATH, 'rb'))
    kmeans = pickle.load(open(KMEANS_MODEL_PATH, 'rb'))
except FileNotFoundError as e:
    print("\nLỗi: Không tìm thấy 1 trong 3 file model.")
    print("Hãy đảm bảo bạn đã chạy 'feature_extraction.py' (bản LSA) và 'clustering.py' (bản LSA) trước.")
    print(f"Chi tiết lỗi: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi khi nạp file: {e}")
    sys.exit(1)

# --- 2. TRÍCH XUẤT TỪ KHÓA (PHƯƠNG PHÁP LSA) ---
print("\nTrích xuất từ khóa cho từng chủ đề (LSA)...")
try:
    # Lấy "cuốn từ điển" (5000 từ) từ TF-IDF
    terms = vectorizer.get_feature_names_out()
    
    # Lấy tâm cụm (centroids) từ K-Means. 
    # Chúng đang ở dạng nén (300 chiều)
    centroids_lsa = kmeans.cluster_centers_

    
    # Lấy mô hình SVD từ bên trong pipeline LSA
    svd = lsa_model.named_steps['truncatedsvd']
    
    # Biến đổi ngược các tâm cụm 300 chiều về lại không gian TF-IDF 5000 chiều
    centroids_tfidf = svd.inverse_transform(centroids_lsa)

    # Sắp xếp các từ khóa dựa trên trọng số ở không gian 5000 chiều
    sorted_term_indices = centroids_tfidf.argsort()[:, ::-1]

    print(f"\n--- CÁC CHỦ ĐỀ ĐƯỢC PHÁT HIỆN (K={K_VALUE}, LSA) ---")
    num_keywords = 15
    for i in range(K_VALUE):
        top_keywords_indices = sorted_term_indices[i, :num_keywords]
        top_keywords = [terms[idx] for idx in top_keywords_indices]
        print(f"Chủ đề {i}: {', '.join(top_keywords)}")
        
except Exception as e:
    print(f"Lỗi khi trích xuất từ khóa: {e}")