import pandas as pd
import pickle
import numpy as np
import os

# --- CẤU HÌNH ---
K_VALUE = 34
TOP_KEYWORDS = 15
TOP_DOCS = 3
DATA_DIR = 'data/processed'

# --- 1. NẠP DỮ LIỆU ---
print("Đang nạp dữ liệu...")
vectorizer = pickle.load(open(f'{DATA_DIR}/lsa_tfidf_vectorizer.pkl', 'rb'))
lsa_model = pickle.load(open(f'{DATA_DIR}/lsa_model.pkl', 'rb'))
kmeans = pickle.load(open(f'{DATA_DIR}/two_stage_results/kmeans_two_stage.pkl', 'rb'))
with open(f'{DATA_DIR}/lsa_matrix.pkl', 'rb') as f:
    X_lsa = pickle.load(f)

df = pd.read_csv(f'{DATA_DIR}/processed_data.csv')
# Lọc bỏ dữ liệu rỗng ngay từ đầu
df = df.dropna(subset=['processed_content'])
df = df[df['processed_content'].str.strip() != ''].reset_index(drop=True)

# --- 2. XỬ LÝ ---
# Gán cụm và tính khoảng cách tới tâm cụm (lấy giá trị min của khoảng cách)
df['cluster'] = kmeans.predict(X_lsa)
df['distance'] = kmeans.transform(X_lsa).min(axis=1)

# Giải mã từ khóa (LSA -> TF-IDF -> Terms)
terms = vectorizer.get_feature_names_out()
centroids_tfidf = lsa_model.named_steps['truncatedsvd'].inverse_transform(kmeans.cluster_centers_)
sorted_keywords_idx = centroids_tfidf.argsort()[:, ::-1]

# --- 3. HIỂN THỊ KẾT QUẢ ---
print(f"\n--- KẾT QUẢ PHÂN TÍCH ({K_VALUE} CHỦ ĐỀ) ---")

for k in range(K_VALUE):
    print(f"\n=== CHỦ ĐỀ {k} ===")
    
    # A. In từ khóa
    keywords = [terms[i] for i in sorted_keywords_idx[k, :TOP_KEYWORDS]]
    print(f"  TỪ KHÓA: {', '.join(keywords)}")
    
    # B. Tìm bài báo đại diện (Sắp xếp theo distance tăng dần -> Lọc tiêu đề rác -> Lấy top đầu)
    cluster_docs = df[df['cluster'] == k].sort_values('distance')
    
    # Lọc bài có tiêu đề hợp lệ (dài hơn 10 ký tự và không phải "không tìm thấy")
    valid_docs = cluster_docs[
        (cluster_docs['title'].str.len() > 10) & 
        (cluster_docs['title'].str.lower() != 'không tìm thấy')
    ]
    
    # In top bài báo
    for i, title in enumerate(valid_docs['title'].head(TOP_DOCS)):
        print(f"    #{i+1}: {title}")