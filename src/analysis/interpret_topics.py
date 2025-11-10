import pickle
import sys
import os


print("Nạp các file đã lưu...")
model_path = 'data/processed/clustered_results/kmeans_model_k19.pkl'
vectorizer_path = 'data/processed/tfidf_vectorizer.pkl'

try:
    # Nạp mô hình K-Means
    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)

    # Nạp TfidfVectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{model_path}' hoặc '{vectorizer_path}'.")
    print("Hãy chạy các file clustering.py và feature_extraction.py trước.")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi không xác định khi nạp file: {e}")
    sys.exit(1)


#  Trích xuất từ khóa cho từng chủ đề

print("\nTrích xuất từ khóa cho từng chủ đề...")
try:
    # Lấy danh sách các từ (features) từ vectorizer
    terms = vectorizer.get_feature_names_out()

    # Lấy các vector tâm của mỗi cụm (centroids)
    # Mỗi centroid là một vector đại diện cho trung tâm của cụm
    centroids = kmeans.cluster_centers_

    # Sắp xếp các chỉ số của từ trong mỗi centroid theo giá trị TF-IDF giảm dần
    # argsort() trả về chỉ số của các phần tử sau khi đã sắp xếp
    sorted_term_indices = centroids.argsort()[:, ::-1] # Sắp xếp giảm dần

    print("\n--- CÁC CHỦ ĐỀ ĐƯỢC PHÁT HIỆN ---")
    # Duyệt qua từng cụm để lấy ra các từ khóa top
    num_keywords = 15 # Số lượng từ khóa muốn hiển thị cho mỗi chủ đề
    for i in range(kmeans.n_clusters):
        # Lấy ra N từ khóa hàng đầu cho cụm thứ i
        top_keywords_indices = sorted_term_indices[i, :num_keywords]
        top_keywords = [terms[idx] for idx in top_keywords_indices]

        print(f"Chủ đề {i}: {', '.join(top_keywords)}")

except Exception as e:
    print(f"Lỗi khi trích xuất từ khóa: {e}")