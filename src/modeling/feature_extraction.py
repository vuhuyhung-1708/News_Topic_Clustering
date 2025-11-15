import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import pickle
import sys
import os
import time

# --- CẤU HÌNH ---
N_COMPONENTS = 300   # Số chiều "tinh túy" muốn nén xuống (từ 5000 -> 300)

# --- THIẾT LẬP ĐƯỜNG DẪN ---
# (Sử dụng đường dẫn tương đối đơn giản, giả định chạy từ thư mục gốc)
PROCESSED_DIR = 'data/processed'
DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_data.csv')

# --- THAY ĐỔI TÊN FILE OUTPUT CHO ĐỒNG BỘ ---
# Cả 3 file giờ đây sẽ bắt đầu bằng "lsa_"
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl') # "Từ điển" TF-IDF (dùng cho LSA)
LSA_MODEL_PATH = os.path.join(PROCESSED_DIR, 'lsa_model.pkl')         # "Bộ nén" LSA
LSA_MATRIX_PATH = os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl')      # Ma trận LSA (vector tốt nhất)
# ---------------------------------------------

# --- BƯỚC 1: ĐỌC DỮ LIỆU ĐÃ XỬ LÝ ---
try:
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=['processed_content'], inplace=True)
    df = df[df['processed_content'].str.strip() != '']
    print(f"Đọc dữ liệu đã tiền xử lý thành công từ: '{DATA_PATH}' ({len(df)} bài báo)")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{DATA_PATH}'. Hãy chạy preprocess.py trước.")
    sys.exit(1)
corpus = df['processed_content']

# --- BƯỚC 2: KHỞI TẠO VÀ ÁP DỤNG TF-IDF ---
print("Bắt đầu trích xuất đặc trưng bằng TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
try:
    X_tfidf = vectorizer.fit_transform(corpus)
    print("Trích xuất đặc trưng TF-IDF hoàn tất!")
    print(f"Kích thước ma trận TF-IDF gốc: {X_tfidf.shape}")
except Exception as e:
    print(f"Lỗi khi thực hiện TF-IDF: {e}")
    sys.exit(1)

# --- BƯỚC 3: KẾT HỢP VỚI LSA (TRUNCATEDSVD) ---
print(f"\nĐang thực hiện LSA để nén từ {X_tfidf.shape[1]} xuống {N_COMPONENTS} chiều...")
start_time = time.time()
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
normalizer = Normalizer(copy=False)
lsa_pipeline = make_pipeline(svd, normalizer)
X_lsa = lsa_pipeline.fit_transform(X_tfidf)
explained_variance = svd.explained_variance_ratio_.sum()
print(f"LSA hoàn tất! Kích thước ma trận mới: {X_lsa.shape} (Thời gian: {time.time() - start_time:.2f}s)")
print(f"   -> LSA giữ lại được {explained_variance:.1%} thông tin.")

# --- BƯỚC 4: LƯU CÁC KẾT QUẢ ---
print("\nĐang lưu các file model và ma trận đã xử lý...")
try:
    # 1. Lưu "cuốn từ điển" TF-IDF
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Đã lưu TF-IDF Vectorizer vào: '{VECTORIZER_PATH}'")

    # 2. Lưu "bộ nén" LSA
    with open(LSA_MODEL_PATH, 'wb') as f:
        pickle.dump(lsa_pipeline, f)
    print(f"Đã lưu LSA model vào: '{LSA_MODEL_PATH}'")

    # 3. Lưu "vector tốt nhất"
    with open(LSA_MATRIX_PATH, 'wb') as f:
        pickle.dump(X_lsa, f)
    print(f"Đã lưu ma trận LSA (vector tốt nhất) vào: '{LSA_MATRIX_PATH}'")

except Exception as e:
    print(f"Lỗi khi lưu các file pickle: {e}")

print("\n--- Hoàn thành ---")