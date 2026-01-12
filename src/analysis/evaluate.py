import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

# File Input (Ưu tiên LSA)
LSA_MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl')
TFIDF_MATRIX_PATH = os.path.join(PROCESSED_DIR, 'tfidf_matrix.pkl')
DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')

# File Output
PLOT_SAVE_PATH = os.path.join(PROCESSED_DIR, 'elbow_chart_clean.png')

def main():
    print("--- PHƯƠNG PHÁP KHUỶU TAY (ELBOW METHOD) - CLEAN VERSION ---")
    
    # 1. NẠP DỮ LIỆU
    X = None
    print("1. Đang nạp dữ liệu...")
    
    if os.path.exists(LSA_MATRIX_PATH):
        print(f"   -> ✅ Đã tìm thấy file LSA: {LSA_MATRIX_PATH}")
        with open(LSA_MATRIX_PATH, 'rb') as f:
            X = pickle.load(f)
    elif os.path.exists(TFIDF_MATRIX_PATH):
        print(f"   -> ⚠️ Đang dùng file TF-IDF: {TFIDF_MATRIX_PATH}")
        with open(TFIDF_MATRIX_PATH, 'rb') as f:
            X = pickle.load(f)
    else:
        print("   -> ⚠️ Không tìm thấy file ma trận. Đang tính toán lại từ CSV...")
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            df = df.dropna(subset=['processed_content'])
            corpus = df['processed_content'].tolist()
            vectorizer = TfidfVectorizer(max_features=5000) 
            X = vectorizer.fit_transform(corpus)
        else:
            print(f"❌ Lỗi: Không tìm thấy dữ liệu gốc.")
            return

    print(f"   -> Kích thước ma trận: {X.shape}")

    # 2. TÍNH TOÁN
    print(f"\n2. Đang chạy K-Means từ K={K_MIN} đến K={K_MAX}...")
    inertias = []
    k_range = list(range(K_MIN, K_MAX + 1, STEP))
    
    start_time = time.time()

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            print(f"   -> K={k}: Inertia={kmeans.inertia_:.1f}", end='\r')
        except Exception as e:
            print(f"   -> ❌ Lỗi tại K={k}")
            inertias.append(0)

    print(f"\n⏱️ Hoàn tất trong {time.time() - start_time:.0f} giây!")

    # 3. VẼ BIỂU ĐỒ (Đơn giản hóa)
    print("3. Đang vẽ biểu đồ...")
    plt.figure(figsize=(10, 6))
    
    # Chỉ vẽ một đường duy nhất màu xanh, có chấm tròn
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=6, label='Inertia')
    
    plt.title('Phương pháp Khuỷu tay (Elbow Method)', fontsize=16)
    plt.xlabel('Số lượng cụm (K)', fontsize=12)
    plt.ylabel('Inertia (Tổng bình phương khoảng cách nội cụm)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()
    
    print(f"✅ Đã lưu biểu đồ sạch tại: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()