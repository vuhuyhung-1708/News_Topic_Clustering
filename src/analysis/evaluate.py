import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import sys
import pickle
import numpy as np
import time

# --- CẤU HÌNH ---
K_MIN = 10
K_MAX = 60
STEP = 2

# --- ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Các file Input tiềm năng
TFIDF_MATRIX_PATH = os.path.join(PROCESSED_DIR, 'tfidf_matrix.pkl')
LSA_MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl')
DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')

# File Output (Lưu vào data/processed)
PLOT_SAVE_PATH = os.path.join(PROCESSED_DIR, 'silhouette_chart.png')

def main():
    print("--- ĐÁNH GIÁ SILHOUETTE SCORE (VẼ BIỂU ĐỒ) ---")
    
    # 1. Nạp dữ liệu Ma trận
    X = None
    print("1. Đang tìm và nạp ma trận đầu vào...")
    
    # Ưu tiên 1: Tìm ma trận TF-IDF
    if os.path.exists(TFIDF_MATRIX_PATH):
        print(f"   -> Đã tìm thấy file TF-IDF: {TFIDF_MATRIX_PATH}")
        with open(TFIDF_MATRIX_PATH, 'rb') as f:
            X = pickle.load(f)

    # Ưu tiên 2: Tìm ma trận LSA (như ảnh bạn gửi trước đó)
    elif os.path.exists(LSA_MATRIX_PATH):
        print(f"   -> Đã tìm thấy file LSA: {LSA_MATRIX_PATH}")
        with open(LSA_MATRIX_PATH, 'rb') as f:
            X = pickle.load(f)

    # Fallback: Tính lại từ CSV
    else:
        print("   -> ⚠️ Không tìm thấy file ma trận .pkl. Đang tính toán lại từ CSV...")
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            df = df.dropna(subset=['processed_content'])
            corpus = df['processed_content'].tolist()
            vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
            X = vectorizer.fit_transform(corpus)
        else:
            print(f"Lỗi: Không tìm thấy dữ liệu gốc tại {DATA_PATH}")
            return

    print(f"   -> Kích thước ma trận sử dụng: {X.shape}")

    # 2. Tính toán Silhouette Score
    print(f"\n2. Bắt đầu tính Silhouette từ K={K_MIN} đến K={K_MAX}...")
    silhouette_values = []
    k_range = range(K_MIN, K_MAX + 1, STEP)
    
    start_time_total = time.time()

    for k in k_range:
        try:
            # n_init=3 để chạy nhanh hơn
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            labels = kmeans.fit_predict(X)
            
            # Tính điểm Silhouette
            score = silhouette_score(X, labels)
            silhouette_values.append(score)
            
            print(f"   -> K={k}: Silhouette Score = {score:.4f}")
        except Exception as e:
            print(f"   -> Lỗi tại K={k}: {e}")
            silhouette_values.append(0)

    print(f"\nHoàn tất tính toán trong {time.time() - start_time_total:.0f} giây!")

    # 3. Vẽ và Lưu biểu đồ
    print("3. Đang vẽ biểu đồ Silhouette...")
    
    plt.figure(figsize=(12, 6))
    # Vẽ đường nối các điểm
    plt.plot(k_range, silhouette_values, 'ro-', linewidth=2, markersize=6, label='Silhouette Score')
    
    # Tìm và đánh dấu điểm cao nhất
    if silhouette_values:
        best_idx = np.argmax(silhouette_values)
        best_k = list(k_range)[best_idx]
        best_score = silhouette_values[best_idx]
        
        plt.plot(best_k, best_score, 'b*', markersize=15, label=f'Best K = {best_k}')
        # Vẽ đường gióng xuống trục X
        plt.axvline(x=best_k, color='gray', linestyle='--', alpha=0.5)
        
        print(f"\n🏆 KẾT LUẬN: Số cụm tối ưu theo Silhouette là K = {best_k} (Score = {best_score:.4f})")
    
    plt.title(f'Biểu đồ Silhouette Score theo số cụm K ({K_MIN}–{K_MAX})', fontsize=16)
    plt.xlabel('Số cụm (K)', fontsize=14)
    plt.ylabel('Silhouette Score (Càng cao càng tốt)', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Lưu biểu đồ vào data/processed
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()
    
    print(f"✅ Đã lưu biểu đồ thành công tại: {PLOT_SAVE_PATH}")
    print(f"Hãy kiểm tra thư mục '{PROCESSED_DIR}' để xem ảnh 'silhouette_chart.png'")

if __name__ == "__main__":
    main()