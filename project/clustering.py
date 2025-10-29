import pandas as pd
from sklearn.cluster import KMeans
import pickle
import sys
import os

# --- BƯỚC 1: NẠP DỮ LIỆU ĐÃ TRÍCH XUẤT ĐẶC TRƯNG ---
print("Nạp ma trận TF-IDF và dữ liệu đã xử lý...")
matrix_path = 'data/processed/tfidf_matrix.pkl'
processed_data_path = 'data/processed/processed_data.csv'
output_folder = 'data/processed' # Thư mục lưu kết quả

try:
    # Nạp ma trận TF-IDF
    with open(matrix_path, 'rb') as f:
        X = pickle.load(f)

    # Nạp lại file processed_data để gắn nhãn cụm
    df = pd.read_csv(processed_data_path)
    # Đảm bảo số dòng khớp với ma trận X sau khi đã làm sạch ở bước trước
    df.dropna(subset=['processed_content'], inplace=True)
    df = df[df['processed_content'].str.strip() != '']

    # Kiểm tra khớp số dòng
    if X.shape[0] != len(df):
        print(f"Lỗi: Số dòng của ma trận TF-IDF ({X.shape[0]}) không khớp với số dòng trong file CSV ({len(df)}).")
        print("Vui lòng chạy lại preprocess.py và feature_extraction.py.")
        sys.exit(1)
        
    print(f"Nạp dữ liệu thành công. Số bài báo: {X.shape[0]}")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{matrix_path}' hoặc '{processed_data_path}'.")
    print("Hãy chạy các file preprocess.py và feature_extraction.py trước.")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi không xác định khi nạp dữ liệu: {e}")
    sys.exit(1)

# --- BƯỚC 2: CHẠY THUẬT TOÁN K-MEANS ---
# Bạn cần chọn số cụm (chủ đề) muốn phát hiện.
# Đây là một tham số quan trọng, chúng ta sẽ bắt đầu với k=10.
# Trong các bước nâng cao, bạn sẽ học cách tìm ra k tối ưu.
k = 10 
# n_init='auto' (hoặc >=10) để chạy K-Means nhiều lần với các tâm khởi tạo khác nhau, chọn kết quả tốt nhất.
# random_state để đảm bảo kết quả có thể lặp lại.
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') 


print(f"\nBắt đầu phân cụm {X.shape[0]} bài báo thành {k} chủ đề...")
try:
    # Huấn luyện mô hình trên ma trận TF-IDF X
    kmeans.fit(X)

    # Lấy nhãn cụm cho từng bài báo (các số từ 0 đến k-1)
    labels = kmeans.labels_
    print("Phân cụm hoàn tất!")
except Exception as e:
    print(f"Lỗi khi chạy K-Means: {e}")
    sys.exit(1)

# --- BƯỚC 3: GẮN NHÃN CỤM VÀ LƯU KẾT QUẢ ---
# Thêm cột 'cluster' chứa nhãn cụm vào DataFrame
df['cluster'] = labels

# Lưu DataFrame đã có nhãn cụm vào file CSV mới
clustered_data_path = os.path.join(output_folder, 'clustered_data.csv')
model_path = os.path.join(output_folder, 'kmeans_model.pkl')

try:
    df.to_csv(clustered_data_path, index=False, encoding='utf-8-sig')
    print(f"\nĐã lưu kết quả phân cụm vào file: '{clustered_data_path}'")

    # Lưu lại mô hình kmeans đã huấn luyện để dùng sau
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"Đã lưu mô hình K-Means vào file: '{model_path}'")

    # In ra xem thử số lượng bài báo trong mỗi cụm
    print("\nThống kê số lượng bài báo trong mỗi cụm:")
    print(df['cluster'].value_counts().sort_index()) # Sắp xếp theo chỉ số cụm cho dễ nhìn

except Exception as e:
    print(f"Lỗi khi lưu kết quả: {e}")