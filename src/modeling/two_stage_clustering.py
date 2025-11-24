import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
import pickle
import sys
import os
import numpy as np

# --- CẤU HÌNH ---
# 1. Tham số DBSCAN (để lọc nhiễu)
EPS_VALUE = 0.85
MIN_SAMPLES_VALUE = 5

# 2. Tham số K-Means (để phân cụm dữ liệu sạch)
# Lưu ý: Vì dữ liệu đã ít đi (do lọc nhiễu), bạn có thể giảm K xuống một chút
K_VALUE = 34

# --- ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl')
CSV_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')

# Thư mục lưu kết quả
OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'two_stage_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. NẠP DỮ LIỆU ---
print("--- BẮT ĐẦU QUY TRÌNH PHÂN CỤM 2 GIAI ĐOẠN ---")
print("1. Nạp dữ liệu...")
try:
    with open(MATRIX_PATH, 'rb') as f:
        X = pickle.load(f) # Ma trận LSA đầy đủ
    df = pd.read_csv(CSV_PATH)
    df.dropna(subset=['processed_content'], inplace=True)
    df = df[df['processed_content'].str.strip() != '']
    df.reset_index(drop=True, inplace=True) # Reset index để khớp ma trận

    print(f"   Tổng số bài báo ban đầu: {len(df)}")
except Exception as e:
    print(f"Lỗi nạp dữ liệu: {e}")
    sys.exit(1)

# --- 2. GIAI ĐOẠN 1: LỌC NHIỄU BẰNG DBSCAN ---
print(f"\n2. Chạy DBSCAN để lọc nhiễu (eps={EPS_VALUE}, min_samples={MIN_SAMPLES_VALUE})...")
dbscan = DBSCAN(eps=EPS_VALUE, min_samples=MIN_SAMPLES_VALUE, metric='euclidean')
db_labels = dbscan.fit_predict(X)

# Xác định các điểm dữ liệu KHÔNG phải là nhiễu (nhãn != -1)
# np.where trả về vị trí (index) của các bài báo "sạch"
core_indices = np.where(db_labels != -1)[0]
noise_indices = np.where(db_labels == -1)[0]

print(f"   -> Số bài báo 'Nhiễu' (bị loại bỏ): {len(noise_indices)} ({len(noise_indices)/len(df)*100:.1f}%)")
print(f"   -> Số bài báo 'Cốt lõi' (được giữ lại): {len(core_indices)} ({len(core_indices)/len(df)*100:.1f}%)")

if len(core_indices) < K_VALUE:
    print("Lỗi: Số lượng bài báo còn lại quá ít để chạy K-Means. Hãy tăng eps của DBSCAN.")
    sys.exit(1)

# Tạo bộ dữ liệu mới chỉ chứa các bài báo sạch
X_clean = X[core_indices]
df_clean = df.iloc[core_indices].copy()
df_clean.reset_index(drop=True, inplace=True) # Reset index cho bộ dữ liệu mới

# --- 3. GIAI ĐOẠN 2: PHÂN CỤM TINH CHỈNH BẰNG K-MEANS ---
print(f"\n3. Chạy K-Means trên {len(df_clean)} bài báo sạch (K={K_VALUE})...")
kmeans = KMeans(n_clusters=K_VALUE, random_state=42, n_init='auto')
kmeans_labels = kmeans.fit_predict(X_clean)

# Gán nhãn vào DataFrame sạch
df_clean['cluster'] = kmeans_labels

# --- 4. LƯU KẾT QUẢ ---
print("\n4. Lưu kết quả...")
# Lưu file CSV kết quả (chỉ chứa bài báo sạch)
clean_csv_path = os.path.join(OUTPUT_DIR, 'two_stage_clusters.csv')
df_clean.to_csv(clean_csv_path, index=False, encoding='utf-8-sig')

# Lưu model K-Means mới (để dùng cho việc gán nhãn/dự đoán sau này)
kmeans_model_path = os.path.join(OUTPUT_DIR, 'kmeans_two_stage.pkl')
with open(kmeans_model_path, 'wb') as f:
    pickle.dump(kmeans, f)

# Lưu file chứa các bài báo bị loại bỏ (để tham khảo)
noise_csv_path = os.path.join(OUTPUT_DIR, 'noise_articles.csv')
df.iloc[noise_indices].to_csv(noise_csv_path, index=False, encoding='utf-8-sig')

print(f"✅ Đã lưu kết quả phân cụm sạch vào: {clean_csv_path}")
print(f"✅ Đã lưu danh sách bài báo nhiễu vào: {noise_csv_path}")
print(f"✅ Đã lưu model K-Means vào: {kmeans_model_path}")

print("\n--- THỐNG KÊ KẾT QUẢ CUỐI CÙNG ---")
print(df_clean['cluster'].value_counts().sort_index())