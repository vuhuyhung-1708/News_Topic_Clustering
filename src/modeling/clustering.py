import pandas as pd
from sklearn.cluster import KMeans
import pickle
import sys
import os


K_VALUE = 34


PROCESSED_DIR = 'data/processed'

# 1. ĐƯỜNG DẪN INPUT (Đọc ma trận LSA "tốt nhất")
MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl') 
CSV_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')

# 2. ĐƯỜNG DẪN OUTPUT (Lưu vào thư mục 'clustered_results' như bạn muốn)
OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'clustered_results')
os.makedirs(OUTPUT_DIR, exist_ok=True) # Tự động tạo thư mục nếu chưa có

MODEL_PATH = os.path.join(OUTPUT_DIR, f'kmeans_lsa_k{K_VALUE}.pkl') # Tên file mới
RESULT_PATH = os.path.join(OUTPUT_DIR, f'clustered_lsa_k{K_VALUE}.csv') # Tên file mới
# ----------------------------------------

def load_data(matrix_path, csv_path):
    """Nạp ma trận LSA (đã nén) và file CSV, kiểm tra tính toàn vẹn."""
    print("Nạp ma trận LSA (vector đã nén) và file CSV...")
    try:
        with open(matrix_path, 'rb') as f:
            X = pickle.load(f) # Nạp ma trận LSA
        df = pd.read_csv(csv_path) # Nạp file CSV
        
        df.dropna(subset=['processed_content'], inplace=True)
        df = df[df['processed_content'].str.strip() != '']

        if X.shape[0] != len(df):
            print(f"Lỗi: Số dòng ma trận ({X.shape[0]}) và CSV ({len(df)}) không khớp.")
            sys.exit(1)
            
        print(f"Nạp dữ liệu thành công. Kích thước ma trận nén: {X.shape}")
        return X, df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{matrix_path}' hoặc '{csv_path}'.")
        print("Hãy đảm bảo bạn đã chạy 'feature_extraction.py' (phiên bản nâng cấp LSA) trước.")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi nạp dữ liệu: {e}.")
        sys.exit(1)

def load_or_train_model(model_path, k, X):
    """Kiểm tra nếu model LSA đã tồn tại thì tải, nếu không thì huấn luyện model mới."""
    if os.path.exists(model_path):
        print(f"Đã tìm thấy model LSA (k={k}). Đang tải...")
        with open(model_path, 'rb') as f:
            kmeans = pickle.load(f)
    else:
        print(f"Chưa có model LSA (k={k}). Bắt đầu huấn luyện model K-Means mới...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X) # Huấn luyện model trên ma trận LSA
        print("Huấn luyện model mới thành công!")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(kmeans, f)
            print(f"Đã lưu model LSA mới vào: '{model_path}'")
        except Exception as e:
            print(f"Lỗi khi lưu model: {e}")
    return kmeans

def save_results(df, labels, result_path):
    """Gán nhãn cụm (từ model LSA) vào DataFrame và lưu kết quả."""
    print("Đang gán nhãn cụm và lưu kết quả...")
    try:
        df['cluster'] = labels
        df.to_csv(result_path, index=False, encoding='utf-8-sig')
        
        print(f"\nĐã lưu kết quả (LSA, k={K_VALUE}) vào file: '{result_path}'")
        print("\nThống kê số lượng bài báo trong mỗi cụm:")
        print(df['cluster'].value_counts().sort_index())
    except Exception as e:
        print(f"Lỗi khi lưu kết quả CSV: {e}")

# --- BƯỚC CHÍNH ĐỂ CHẠY CHƯƠNG TRÌNH ---
if __name__ == '__main__':
    """Hàm chính: Tải dữ liệu LSA, huấn luyện/tải model K-Means, gán nhãn và lưu kết quả."""
    
    print(f"--- Bắt đầu quy trình K-Means trên ma trận LSA (k = {K_VALUE}) ---")
    
    # 1. Tải dữ liệu
    X_lsa, df = load_data(MATRIX_PATH, CSV_PATH)
    
    # 2. Huấn luyện hoặc tải model
    kmeans_model = load_or_train_model(MODEL_PATH, K_VALUE, X_lsa)
    
    # 3. Gán nhãn
    cluster_labels = kmeans_model.predict(X_lsa)
    
    # 4. Lưu kết quả
    save_results(df, cluster_labels, RESULT_PATH)