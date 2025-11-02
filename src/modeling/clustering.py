import pandas as pd
from sklearn.cluster import KMeans
import pickle
import sys
import os

# --- BƯỚC 1: CÁC THIẾT LẬP CƠ BẢN ---
K_VALUE = 10 

# --- ĐƯỜNG DẪN ĐÃ ĐƯỢC ĐƠN GIẢN HÓA ---
# Giả định bạn luôn chạy file này từ thư mục gốc News-Topic-Clustering
PROCESSED_DIR = 'data/processed'
MATRIX_PATH = os.path.join(PROCESSED_DIR, 'tfidf_matrix.pkl')
CSV_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')

# Đường dẫn file OUTPUT
OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'clustered_results')
os.makedirs(OUTPUT_DIR, exist_ok=True) # Tự động tạo thư mục nếu chưa có

MODEL_PATH = os.path.join(OUTPUT_DIR, f'kmeans_model_k{K_VALUE}.pkl')
RESULT_PATH = os.path.join(OUTPUT_DIR, f'clustered_data_k{K_VALUE}.csv')
# ----------------------------------------

def load_data(matrix_path, csv_path):
    """Nạp ma trận TF-IDF và file CSV, kiểm tra tính toàn vẹn."""
    print("Nạp ma trận TF-IDF và file CSV...")
    try:
        with open(matrix_path, 'rb') as f:
            X = pickle.load(f) # Nạp ma trận TF-IDF
        df = pd.read_csv(csv_path) # Nạp file CSV
        
        df.dropna(subset=['processed_content'], inplace=True)
        df = df[df['processed_content'].str.strip() != '']

        if X.shape[0] != len(df):
            print(f"Lỗi: Số dòng ma trận ({X.shape[0]}) và CSV ({len(df)}) không khớp.")
            sys.exit(1)
            
        print(f"Nạp dữ liệu thành công. Số bài báo: {X.shape[0]}")
        return X, df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{matrix_path}' hoặc '{csv_path}'.")
        print("Hãy đảm bảo bạn đang chạy script từ thư mục gốc của dự án.")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi nạp dữ liệu: {e}.")
        sys.exit(1)

def load_or_train_model(model_path, k, X):
    """Kiểm tra nếu model đã tồn tại thì tải, nếu không thì huấn luyện model mới."""
    if os.path.exists(model_path):
        print(f"Đã tìm thấy model đã huấn luyện (k={k}). Đang tải...")
        with open(model_path, 'rb') as f:
            kmeans = pickle.load(f)
    else:
        print(f"Chưa có model (k={k}). Bắt đầu huấn luyện model mới...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X) # Huấn luyện model
        print("Huấn luyện model mới thành công!")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(kmeans, f)
            print(f"Đã lưu model mới vào: '{model_path}'")
        except Exception as e:
            print(f"Lỗi khi lưu model: {e}")
    return kmeans

def save_results(df, labels, result_path):
    """Gán nhãn cụm vào DataFrame và lưu kết quả ra file CSV."""
    print("Đang gán nhãn cụm và lưu kết quả...")
    try:
        df['cluster'] = labels
        df.to_csv(result_path, index=False, encoding='utf-8-sig')
        
        print(f"\nĐã lưu kết quả phân cụm (k={K_VALUE}) vào file: '{result_path}'")
        print("\nThống kê số lượng bài báo trong mỗi cụm:")
        print(df['cluster'].value_counts().sort_index())
    except Exception as e:
        print(f"Lỗi khi lưu kết quả CSV: {e}")

# --- BƯỚC CHÍNH ĐỂ CHẠY CHƯƠNG TRÌNH ---
if __name__ == '__main__':
    """Hàm chính: Tải dữ liệu, huấn luyện/tải model, gán nhãn và lưu kết quả."""
    
    print(f"--- Bắt đầu quy trình cho k = {K_VALUE} ---")
    
    # 1. Tải dữ liệu
    X, df = load_data(MATRIX_PATH, CSV_PATH)
    
    # 2. Huấn luyện hoặc tải model
    kmeans_model = load_or_train_model(MODEL_PATH, K_VALUE, X)
    
    # 3. Gán nhãn
    cluster_labels = kmeans_model.predict(X)
    
    # 4. Lưu kết quả
    save_results(df, cluster_labels, RESULT_PATH)