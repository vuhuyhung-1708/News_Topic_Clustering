import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pickle
import sys
import os
import numpy as np

# --- CẤU HÌNH ---
# Điều chỉnh tham số DBSCAN
EPS_VALUE = 0.9         
MIN_SAMPLES_VALUE = 5   

# Thiết lập đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl')
CSV_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')

# --- CẬP NHẬT ĐƯỜNG DẪN LƯU KẾT QUẢ THEO YÊU CẦU ---
# Tạo thư mục tên là 'dbscan_clustered_results' nằm trong 'data/processed'
OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'dbscan_clustered_results')
os.makedirs(OUTPUT_DIR, exist_ok=True) # Tự tạo folder nếu chưa có

# Tên file đầu ra
OUTPUT_CSV_NAME = 'dbscan_clustered_results.csv'
OUTPUT_MODEL_NAME = 'dbscan_model.pkl'
# --- 1. NẠP DỮ LIỆU ---
print(f"--- [DBSCAN] LƯU KẾT QUẢ TẠI: {OUTPUT_DIR} ---")
print("1. Đang nạp dữ liệu...")
try:
    with open(MATRIX_PATH, 'rb') as f:
        X = pickle.load(f) 
    
    df = pd.read_csv(CSV_PATH)
    df = df[df['processed_content'].notna() & (df['processed_content'].str.strip() != '')]
    df.reset_index(drop=True, inplace=True)
    
    print(f"   -> Đã nạp {len(df)} bài báo.")
except Exception as e:
    print(f"Lỗi nạp dữ liệu: {e}")
    sys.exit(1)

# --- 2. CHẠY DBSCAN ---
print(f"\n2. Đang chạy DBSCAN (eps={EPS_VALUE}, min_samples={MIN_SAMPLES_VALUE})...")
dbscan = DBSCAN(eps=EPS_VALUE, min_samples=MIN_SAMPLES_VALUE, metric='euclidean')
labels = dbscan.fit_predict(X)

# --- 3. ĐÁNH GIÁ SƠ BỘ ---
n_noise = list(labels).count(-1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f"   -> Số cụm tìm được: {n_clusters}")
print(f"   -> Số bài báo nhiễu (Noise -1): {n_noise}")

if n_clusters > 1:
    score = silhouette_score(X, labels)
    print(f"   -> Silhouette Score: {score:.4f}")
else:
    print("   -> Không đủ cụm để tính Silhouette Score.")

# --- 4. LƯU KẾT QUẢ VÀO FOLDER 'dbscan_clustered_results' ---
print("\n3. Lưu kết quả...")

# 4.1. Lưu file CSV
df['cluster'] = labels
csv_save_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
print(f"   ✅ Đã lưu CSV: {csv_save_path}")

# 4.2. Lưu file Model (.model)
model_save_path = os.path.join(OUTPUT_DIR, OUTPUT_MODEL_NAME)
with open(model_save_path, 'wb') as f:
    pickle.dump(dbscan, f)
print(f"   ✅ Đã lưu Model: {model_save_path}")

print("\n--- HOÀN TẤT ---")