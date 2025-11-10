import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time


K_MIN = 5
K_MAX = 40      
STEP = 1

# Thiết lập đường dẫn (đi ngược 3 cấp từ file này ra thư mục gốc)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATRIX_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'tfidf_matrix.pkl')
FIGURES_DIR = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- 1. NẠP DỮ LIỆU ---
print(f"Đang nạp ma trận TF-IDF từ '{MATRIX_PATH}'...")
try:
    with open(MATRIX_PATH, 'rb') as f:
        X = pickle.load(f)
    print(f"Nạp thành công! Kích thước dữ liệu: {X.shape}")
except Exception as e:
    print(f"Lỗi khi nạp file: {e}")
    sys.exit(1)

# --- 2. CHẠY VÒNG LẶP ĐÁNH GIÁ ---
print(f"\nBắt đầu đánh giá từ K={K_MIN} đến K={K_MAX}...")
print("Quá trình này có thể mất vài phút, vui lòng đợi...")
inertia_values = []
silhouette_values = []
k_range = range(K_MIN, K_MAX + 1, STEP)

start_time_total = time.time()

for k in k_range:
    start_time = time.time()
    print(f"Đang chạy K = {k:2d}...", end=" ")
    
    # Huấn luyện K-Means
    # n_init=3 để chạy nhanh hơn. Tăng lên 10 nếu muốn kết quả ổn định hơn nữa.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    labels = kmeans.fit_predict(X)
    
    # Lưu các chỉ số
    inertia_values.append(kmeans.inertia_)
    score = silhouette_score(X, labels, random_state=42) # Có thể thêm sample_size=2000 nếu quá chậm
    silhouette_values.append(score)
    
    print(f"Xong. Silhouette: {score:.4f} | Inertia: {kmeans.inertia_:.0f} ({time.time() - start_time:.1f}s)")

print(f"\nHoàn tất đánh giá trong {time.time() - start_time_total:.0f} giây!")

# --- 3. VẼ VÀ LƯU BIỂU ĐỒ ---
print("Đang vẽ và lưu biểu đồ...")

# Biểu đồ 1: Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, 'bo-', markersize=8, linewidth=2)
plt.title(f'Phương pháp Khuỷu tay (Elbow Method) với K từ {K_MIN}-{K_MAX}')
plt.xlabel('Số cụm (K)')
plt.ylabel('Inertia (Độ nén)')
plt.grid(True)
elbow_path = os.path.join(FIGURES_DIR, 'elbow_method.png')
plt.savefig(elbow_path)
plt.close()

# Biểu đồ 2: Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_values, 'rs-', markersize=8, linewidth=2)
plt.title(f'Chỉ số Silhouette với K từ {K_MIN}-{K_MAX}')
plt.xlabel('Số cụm (K)')
plt.ylabel('Silhouette Score (Càng cao càng tốt)')
plt.grid(True)
# Đánh dấu điểm cao nhất
best_k_idx = np.argmax(silhouette_values)
best_k = k_range[best_k_idx]
best_score = silhouette_values[best_k_idx]
plt.axvline(x=best_k, color='g', linestyle='--', label=f'Best K = {best_k}')
plt.legend()

silhouette_path = os.path.join(FIGURES_DIR, 'silhouette_score.png')
plt.savefig(silhouette_path)
plt.close()

print(f"\nĐã lưu biểu đồ Elbow tại: {elbow_path}")
print(f"\nĐã lưu biểu đồ Silhouette tại: {silhouette_path}")
print(f"\nKẾT LUẬN: Theo chỉ số Silhouette, số cụm tối ưu nhất là K = {best_k} (Điểm: {best_score:.4f})")