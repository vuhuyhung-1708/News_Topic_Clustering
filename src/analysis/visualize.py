import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import os
import math

# --- CẤU HÌNH ---
K_VALUE = 35 # Số cụm tốt nhất bạn đã chọn

# Thiết lập đường dẫn (đi ngược 3 cấp từ file này ra thư mục gốc)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Các đường dẫn file input
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(PROCESSED_DIR, 'clustered_results')
MODEL_PATH = os.path.join(MODEL_DIR, f'kmeans_model_k{K_VALUE}.pkl')
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, 'tfidf_vectorizer.pkl')
FONT_PATH = os.path.join(BASE_DIR, 'src', 'preprocessing','assets', 'NotoSans-Regular.ttf')

# Đường dẫn file output (ảnh)
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'wordclouds', f'k_{K_VALUE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. KIỂM TRA VÀ NẠP DỮ LIỆU ---
print(f"--- Bắt đầu tạo Word Cloud cho K={K_VALUE} ---")

if not os.path.exists(FONT_PATH):
    print(f"Lỗi: Không tìm thấy file font tại '{FONT_PATH}'.")
    print("Vui lòng tải font tiếng Việt (.ttf), đổi tên thành 'NotoSans-Regular.ttf' và đặt vào thư mục 'src/assets/font/'.")
    sys.exit(1)

try:
    print("Đang nạp model và vectorizer...")
    with open(MODEL_PATH, 'rb') as f:
        kmeans = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file model hoặc vectorizer.")
    print(f"Đảm bảo bạn đã chạy clustering.py với K={K_VALUE} trước.")
    sys.exit(1)

# --- 2. TẠO VÀ LƯU WORD CLOUD ---
print(f"Đang tạo {K_VALUE} ảnh Word Cloud và lưu vào '{OUTPUT_DIR}'...")
terms = vectorizer.get_feature_names_out()
centroids = kmeans.cluster_centers_
sorted_term_indices = centroids.argsort()[:, ::-1]

for i in range(K_VALUE):
    print(f"  - Đang xử lý Chủ đề {i}...", end=" ")
    try:
        # Lấy 50 từ khóa quan trọng nhất và điểm số TF-IDF của chúng
        top_indices = sorted_term_indices[i, :30]
        keywords_scores = {terms[idx]: centroids[i, idx] for idx in top_indices}

        # Tạo WordCloud
        wc = WordCloud(
            background_color='white',
            max_words=50,
            width=800, height=400,
            font_path=FONT_PATH # Quan trọng để hiển thị tiếng Việt
        ).generate_from_frequencies(keywords_scores)

        # Lưu trực tiếp thành file ảnh
        image_path = os.path.join(OUTPUT_DIR, f'topic_{i}.png')
        wc.to_file(image_path)
        print("Xong.")

    except Exception as e:
        print(f"\nLỗi khi tạo chủ đề {i}: {e}")

print("\nHoàn tất! Hãy kiểm tra thư mục 'results/wordclouds/'.")