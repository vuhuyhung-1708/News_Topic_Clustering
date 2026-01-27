import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import pickle

# --- CẤU HÌNH ---
K_MIN = 10
K_MAX = 60
STEP = 2

# --- ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

LSA_MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl')
DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')
PLOT_SAVE_PATH = os.path.join(PROCESSED_DIR, 'elbow_chart_clean.png')

print("\n--- ELBOW METHOD (CLEAN VERSION) ---")

# --- 1. NẠP DỮ LIỆU ---
if os.path.exists(LSA_MATRIX_PATH):
    print("✔ Sử dụng ma trận LSA")
    X = pickle.load(open(LSA_MATRIX_PATH, 'rb'))


print(f"✔ Kích thước dữ liệu: {X.shape}")

# --- 2. CHẠY K-MEANS ---
print(f"\n▶ Chạy K-Means từ K={K_MIN} → {K_MAX}")

k_values = range(K_MIN, K_MAX + 1, STEP)
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    print(f"   K={k}, Inertia={kmeans.inertia_:.1f}", end='\r')


# --- 3. VẼ BIỂU ĐỒ ---
print("▶ Vẽ biểu đồ Elbow")
plt.figure(figsize=(10, 6))
plt.plot(list(k_values), inertias, marker='o')
plt.xlabel("Số cụm (K)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method")
plt.grid(True)
plt.savefig(PLOT_SAVE_PATH)
plt.close()

print(f"✔ Đã lưu biểu đồ tại: {PLOT_SAVE_PATH}")
print("--- HOÀN THÀNH ---")
