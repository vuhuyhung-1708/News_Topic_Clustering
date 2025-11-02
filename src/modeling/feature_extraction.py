import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
import os

processed_data_path = 'data/processed/processed_data.csv'
try:
    df = pd.read_csv(processed_data_path)
    # Kiểm tra và xử lý các dòng có thể bị rỗng sau khi tiền xử lý
    df.dropna(subset=['processed_content'], inplace=True)
    df = df[df['processed_content'].str.strip() != ''] # Đảm bảo không có chuỗi rỗng

    if df.empty:
        print(f"Lỗi: File '{processed_data_path}' không có dữ liệu hợp lệ sau khi làm sạch.")
        sys.exit(1)

    print(f"Đọc dữ liệu đã tiền xử lý thành công từ: '{processed_data_path}' ({len(df)} bài báo)")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{processed_data_path}'. Hãy chạy file preprocess.py trước.")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi không xác định khi đọc file CSV: {e}")
    sys.exit(1)

corpus = df['processed_content']

# KHỞI TẠO VÀ ÁP DỤNG TF-IDF 

print("Bắt đầu trích xuất đặc trưng bằng TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)

try:
    # .fit_transform() sẽ học từ vựng từ 'corpus' và chuyển đổi 'corpus' thành ma trận.
    X = vectorizer.fit_transform(corpus)
    print("Trích xuất đặc trưng hoàn tất!")
except Exception as e:
    print(f"Lỗi khi thực hiện TF-IDF: {e}")
    sys.exit(1)


# kiểm tra và lưu kết quả 
print(f"Kích thước của ma trận TF-IDF (số bài báo, số từ khóa): {X.shape}")

output_folder = 'data/processed'
os.makedirs(output_folder, exist_ok=True)

# Lưu ma trận X và vectorizer để sử dụng trong bước tiếp theo
matrix_path = os.path.join(output_folder, 'tfidf_matrix.pkl')
vectorizer_path = os.path.join(output_folder, 'tfidf_vectorizer.pkl')

try:
    with open(matrix_path, 'wb') as f:
        pickle.dump(X, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Đã lưu ma trận TF-IDF vào: '{matrix_path}'")
    print(f"Đã lưu TfidfVectorizer vào: '{vectorizer_path}'")
except Exception as e:
    print(f"Lỗi khi lưu các file pickle: {e}")