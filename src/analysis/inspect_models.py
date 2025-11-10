import pickle
import sys
import os

model_path = 'data/processed/clustered_results/kmeans_model_k19.pkl'
vectorizer_path = 'data/processed/tfidf_vectorizer.pkl'

print(f"--- 1. Kiểm tra file K-Means Model ({model_path}) ---")
try:
    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)
    
    print("  Tải file thành công!")
    print(f"  Loại đối tượng: {type(kmeans)}")
    
    # Lấy ra các tâm cụm (centroids)
    centroids = kmeans.cluster_centers_
    print(f"  Model này đã học được {centroids.shape[0]} tâm cụm (chủ đề).")
    print(f"  Mỗi tâm cụm được đại diện bởi một vector có {centroids.shape[1]} chiều (từ khóa).")
    # In ra vị trí của tâm cụm đầu tiên 
except FileNotFoundError:
    print(f"  Lỗi: Không tìm thấy file. Hãy chạy clustering.py trước.")
except Exception as e:
    print(f"  Lỗi khi đọc file: {e}")


print(f"\n--- 2. Kiểm tra file TF-IDF Vectorizer ({vectorizer_path}) ---")
try:
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    print("  Tải file thành công!")
    print(f"  Loại đối tượng: {type(vectorizer)}")
    
    vocabulary = vectorizer.get_feature_names_out()
    print(f"  Vectorizer này đã học được một bộ từ điển gồm {len(vocabulary)} từ.")
    
    print("\n  50 từ đầu tiên trong từ điển (ví dụ):")
    print(vocabulary[:50])

except FileNotFoundError:
    print(f"  Lỗi: Không tìm thấy file. Hãy chạy feature_extraction.py trước.")
except Exception as e:
    print(f"  Lỗi khi đọc file: {e}")