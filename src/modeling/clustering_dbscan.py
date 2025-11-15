import pandas as pd
from sklearn.cluster import DBSCAN
import pickle
import sys
import os
import numpy as np

# --- CẤU HÌNH ---
# Đây là 2 tham số quan trọng nhất của DBSCAN
# Bạn sẽ cần "thử nghiệm" (thay đổi) 2 giá trị này để xem kết quả nào tốt nhất
EPS_VALUE = 0.5       # Khoảng cách tối đa để coi 2 điểm là "hàng xóm"
MIN_SAMPLES_VALUE = 5 # Số lượng "hàng xóm" tối thiểu để tạo thành một cụm

# --- ĐƯỜNG DẪN ĐÃ ĐƯỢC ĐƠN GIẢN HÓA ---
PROCESSED_DIR = 'data/processed'
MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl')
CSV_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')

# Đường dẫn file OUTPUT
OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'clustered_results_dbscan')
os.makedirs(OUTPUT_DIR, exist_ok=True) 
RESULT_PATH = os.path.join(OUTPUT_DIR, f'dbscan_eps{EPS_VALUE}_min{MIN_SAMPLES_VALUE}.csv')

# --- 1. NẠP DỮ LIỆU ---
print("Nạp ma trận LSA (vector đã nén) và file CSV...")
try:
    with open(MATRIX_PATH, 'rb') as f:
        X = pickle.load(f) # Nạp ma trận LSA
    df = pd.read_csv(CSV_PATH) # Nạp file CSV
    
    df.dropna(subset=['processed_content'], inplace=True)
    df = df[df['processed_content'].str.strip() != '']

    if X.shape[0] != len(df):
        print(f"Lỗi: Số dòng ma trận ({X.shape[0]}) và CSV ({len(df)}) không khớp.")
        sys.exit(1)
        
    print(f"Nạp dữ liệu thành công. Kích thước ma trận nén: {X.shape}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại '{MATRIX_PATH}' hoặc '{CSV_PATH}'.")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi khi nạp dữ liệu: {e}.")
    sys.exit(1)

# --- 2. CHẠY THUẬT TOÁN DBSCAN ---
print(f"\nĐang chạy DBSCAN với eps={EPS_VALUE} và min_samples={MIN_SAMPLES_VALUE}...")
# metric='cosine' thường dùng cho dữ liệu văn bản, nhưng 'euclidean' cũng ổn với LSA
dbscan = DBSCAN(eps=EPS_VALUE, min_samples=MIN_SAMPLES_VALUE, metric='euclidean')

try:
    labels = dbscan.fit_predict(X)
    print("Phân cụm DBSCAN hoàn tất!")
except Exception as e:
    print(f"Lỗi khi chạy DBSCAN: {e}")
    sys.exit(1)

# --- 3. PHÂN TÍCH VÀ LƯU KẾT QUẢ ---
print("\n--- KẾT QUẢ PHÂN TÍCH DBSCAN ---")
# Lấy ra các nhãn cụm duy nhất. Nhãn -1 là "nhiễu" (noise/outliers).
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print(f"Số cụm (chủ đề) tự động tìm thấy: {n_clusters}")
print(f"Số bài báo bị coi là 'nhiễu' (không thuộc chủ đề nào): {n_noise} / {len(df)} bài")

try:
    df['cluster'] = labels
    df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\nĐã lưu kết quả DBSCAN vào file: '{RESULT_PATH}'")
    print("\nThống kê số lượng bài báo trong mỗi cụm (nhãn -1 là nhiễu):")
    print(df['cluster'].value_counts())
    
except Exception as e:
    print(f"Lỗi khi lưu kết quả CSV: {e}")