import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hot News Detection", page_icon="🔥", layout="wide")

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .top-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border-left: 8px solid #d32f2f;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        transition: transform 0.2s;
    }
    .top-card:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0,0,0,0.15); }
    .top-rank { font-size: 24px; font-weight: 900; color: #d32f2f; text-transform: uppercase; margin-bottom: 8px; }
    .top-title { font-size: 26px; font-weight: bold; color: #1a237e; margin: 12px 0; line-height: 1.3; }
    .keywords-box { background-color: #f1f8e9; border: 1px solid #c5e1a5; color: #33691e; padding: 10px 15px; border-radius: 8px; font-size: 15px; margin-top: 15px; display: inline-block; }
    .hot-badge { background-color: #ffebee; color: #b71c1c; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: bold; border: 1px solid #ffcdd2; display: inline-block; margin-top: 15px; }
    .related-item { padding: 8px 0; border-bottom: 1px solid #eee; font-size: 16px; }
    
    a { text-decoration: none; color: #1565C0; }
    a:hover { text-decoration: underline; color: #d32f2f; }
</style>
""", unsafe_allow_html=True)

# --- ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TWO_STAGE_DIR = os.path.join(PROCESSED_DIR, 'kmeans_clustered_results')

# --- BẢN ĐỒ TÊN CHỦ ĐỀ (GIỮ NGUYÊN) ---
TOPIC_NAMES = {
    1:  "Sách & Đọc",
    3:  "Đời sống – Văn hóa",
    18: "Công nghệ – Doanh nghiệp",
    9:  "Chính trị – Phát triển",
    2:  "Ẩm thực – Đô thị",
    5:  "Ô tô – Xe máy",
    10: "Pháp luật – Hình sự",
    17: "Gia đình – Hôn nhân",
    6:  "Bóng đá quốc tế",
    23: "Sức khỏe – Dinh dưỡng",
    26: "Ngân hàng – Chứng khoán",
    7:  "Giáo dục – Học đường",
    19: "Y tế – Bệnh viện",
    4:  "Du lịch – Trải nghiệm",
    21: "Điện ảnh – Giải trí",
    20: "Điện thoại – Công nghệ",
    8:  "Thời sự quốc tế",
    27: "Bóng đá châu Á",
    11: "Âm nhạc – Nghệ sĩ",
    16: "Xung đột – Địa chính trị",
    24: "Giao thông – Tai nạn",
    0:  "Giá vàng – Thị trường",
    25: "An toàn thực phẩm",
    12: "Tuyển sinh – Thi cử",
    15: "Thời trang – Sao",
    13: "Chiến sự Ukraine",
    14: "Tham nhũng – Xét xử",
    22: "Truyện Kiều"
}


# --- 1. HÀM NẠP DỮ LIỆU & MODEL ---
@st.cache_resource
def load_data_and_models():
    try:
        vec = pickle.load(open(os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl'), 'rb'))
        lsa = pickle.load(open(os.path.join(PROCESSED_DIR, 'lsa_model.pkl'), 'rb'))
        
        # Đọc model K=34
        km = pickle.load(open(os.path.join(TWO_STAGE_DIR, 'kmeans_model_k22.pkl'), 'rb'))
        
        with open(os.path.join(PROCESSED_DIR, 'lsa_matrix.pkl'), 'rb') as f:
            lsa_matrix = pickle.load(f)

        # Đọc file CSV K=34
        df_path = os.path.join(TWO_STAGE_DIR, 'kmeans_clusters_k22.csv')
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            
            # Chuẩn hóa tên cột cluster (nếu file cũ dùng 'cluster')
            if 'kmeans_cluster' not in df.columns and 'cluster' in df.columns:
                df = df.rename(columns={'cluster': 'kmeans_cluster'})

            df = df.dropna(subset=['processed_content'])
            df = df[df['processed_content'].str.strip() != '']
            df.reset_index(drop=True, inplace=True)
        else:
            df = None
        
        return vec, lsa, km, df, lsa_matrix
    except Exception as e:
        st.error(f"Lỗi nạp dữ liệu: {e}")
        return None, None, None, None, None


# --- 2. HÀM TÍNH TOÁN TRENDING (ĐÃ SỬA LỖI "KHÔNG TÌM THẤY") ---
# --- HÀM KIỂM TRA BÀI BÁO HỢP LỆ (HELPER FUNCTION) ---
def is_valid_article(title, url):
    """Kiểm tra xem bài báo có hợp lệ để hiển thị không."""
    title = str(title).strip()
    url = str(url).strip()
    title_lower = title.lower()
    
    # Danh sách từ khóa lỗi
    invalid_keywords = {'nan', 'none', '', 'null', 'không tìm thấy', 'khong tim thay', '404', 'error'}
    
    # Các điều kiện kiểm tra
    if title_lower in invalid_keywords: return False
    if len(title) < 15: return False
    if len(url) < 10: return False
    
    return True

# --- 2. HÀM TÍNH TOÁN TRENDING (ĐÃ ĐƠN GIẢN HÓA) ---
def get_trending_topics(df, matrix, kmeans, vectorizer, lsa_model):
    if df is None: return []
    
    # 1. Tính toán trước các thông số cần thiết
    distances = kmeans.transform(matrix)
    terms = vectorizer.get_feature_names_out()
    
    # Lấy thành phần SVD để giải mã từ khóa
    svd = lsa_model.named_steps['truncatedsvd'] if hasattr(lsa_model, 'named_steps') else lsa_model
    original_centroids = svd.inverse_transform(kmeans.cluster_centers_)
    ordered_centroids = original_centroids.argsort()[:, ::-1]
    
    # Lấy Top 10 cụm lớn nhất
    top_clusters = df['kmeans_cluster'].value_counts().sort_values(ascending=False).head(10)
    
    trending_list = []
    
    for cluster_id, count in top_clusters.items():
        # 2. Lấy từ khóa (Keywords)
        keywords = [terms[i].replace("_", " ") for i in ordered_centroids[cluster_id, :20]]
        
        # 3. Lấy và sắp xếp các bài báo trong cụm theo khoảng cách đến tâm
        indices = df.index[df['kmeans_cluster'] == cluster_id].tolist()
        cluster_dists = distances[indices, cluster_id]
        # Lấy index thực của 100 bài gần tâm nhất
        sorted_indices = [indices[i] for i in np.argsort(cluster_dists)[:100]]
        
        # 4. Tìm bài đại diện và bài liên quan
        valid_articles = []
        seen_titles = set() # Dùng để khử trùng lặp tiêu đề
        
        for idx in sorted_indices:
            row = df.loc[idx]
            title = row.get('title', '')
            url = row.get('url', '#')
            
            if is_valid_article(title, url) and title not in seen_titles:
                valid_articles.append({'title': title, 'url': url})
                seen_titles.add(title)
            
            # Chỉ cần tìm đủ 11 bài (1 đại diện + 10 liên quan) là dừng
            if len(valid_articles) >= 11:
                break
        
        # 5. Đóng gói kết quả (nếu tìm được ít nhất 1 bài)
        if valid_articles:
            trending_list.append({
                'id': cluster_id,
                'count': count,
                'rep_title': valid_articles[0]['title'], # Bài gần nhất làm đại diện
                'rep_link': valid_articles[0]['url'],
                'keywords': ", ".join(keywords),
                'related': valid_articles[1:] # Các bài còn lại làm bài liên quan
            })
            
    return trending_list

# --- LOAD DỮ LIỆU ---
vectorizer, lsa_model, kmeans, df_data, lsa_matrix = load_data_and_models()

# --- 3. GIAO DIỆN CHÍNH ---

# Lấy dữ liệu trending
trending_topics = []
if df_data is not None:
    # Gọi hàm vừa định nghĩa ở trên
    trending_topics = get_trending_topics(df_data, lsa_matrix, kmeans, vectorizer, lsa_model)

# === SIDEBAR: THÔNG TIN CHUNG ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=80)
    st.title("Hot News Detection")
    st.info(f"Mô hình phân cụm: K-Means (K=34)")
    
    if df_data is not None:
        st.metric(label="Tổng số bài báo", value=f"{len(df_data):,}")
    
    st.markdown("---")
    st.caption("Sinh viên: Vũ Huy Hưng")
    st.caption("Đồ án Tốt nghiệp")

# === MAIN CONTENT: TOP TRENDING ===
st.markdown("<h1 style='text-align: center; color: #d32f2f;'>🔥 CÁC CHỦ ĐỀ TIN TỨC NỔI BẬT NHẤT 🔥</h1>", unsafe_allow_html=True)
if df_data is not None:
    st.markdown(f"<p style='text-align: center; font-size: 18px; color: #555;'>Dữ liệu được cập nhật và phân tích tự động</p>", unsafe_allow_html=True)
st.markdown("---")

col_spacer1, col_content, col_spacer2 = st.columns([1, 6, 1])

with col_content:
    if trending_topics:
        for i, topic in enumerate(trending_topics):
            # Lấy tên chủ đề từ Map
            topic_real_name = TOPIC_NAMES.get(topic['id'], f"Chủ đề {topic['id']}")
            
            st.markdown(f"""
            <div class="top-card">
                <div class="top-rank">TOP {i+1} &nbsp; {topic_real_name}</div>
                <div class="top-title">
                    <a href="{topic['rep_link']}" target="_blank" style="text-decoration:none; color: #1565C0;">
                        "{topic['rep_title']}"
                    </a>
                </div>
                <div class="keywords-box">
                    🔑 <b>Từ khóa chính:</b> {topic['keywords']}
                </div>
                <div style="margin-top:15px; margin-bottom:15px;">
                    <span class="hot-badge">🔥 Mức độ quan tâm: {topic['count']} bài báo</span>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📄 Xem thêm 10 bài báo tiêu biểu khác về chủ đề này"):
                st.markdown('<div class="related-list">', unsafe_allow_html=True)
                if topic['related']:
                    for article in topic['related']:
                        st.markdown(f"""
                        <div class="related-item">
                            • <a href="{article['url']}" target="_blank">{article['title']}</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("Chưa có thêm bài báo liên quan.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("") 
    else:
        st.error("Không tìm thấy dữ liệu. Vui lòng kiểm tra lại thư mục data.")

# Footer
st.markdown("---")
st.caption("Đồ án Tốt nghiệp | GVHD: TS. Nguyễn Mạnh Hiển")