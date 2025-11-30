import streamlit as st
import pandas as pd
import pickle
import os
import re
import numpy as np
from underthesea import word_tokenize

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hot News Detection", page_icon="🔥", layout="wide")

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .top-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #ff4b4b;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        transition: transform 0.2s;
    }
    .top-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
    }
    .top-rank {
        font-size: 20px; 
        font-weight: 800; 
        color: #d32f2f;
        text-transform: uppercase;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    .top-title {
        font-size: 24px;
        font-weight: bold;
        color: #1a237e;
        margin: 12px 0;
        line-height: 1.3;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .keywords-box {
        background-color: #f1f8e9;
        border: 1px solid #c5e1a5;
        color: #33691e;
        padding: 8px 12px;
        border-radius: 5px;
        font-size: 14px;
        margin-top: 10px;
        display: inline-block;
    }
    .hot-badge {
        background-color: #ffebee;
        color: #b71c1c;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        border: 1px solid #ffcdd2;
        display: inline-block;
        margin-top: 15px;
    }
    /* Style cho danh sách bài báo liên quan */
    .related-list {
        margin-top: 10px;
        padding-left: 10px;
    }
    .related-item {
        padding: 6px 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 15px;
    }
    .related-item:last-child {
        border-bottom: none;
    }
    a { text-decoration: none; color: #1565C0; }
    a:hover { text-decoration: underline; color: #d32f2f; }
</style>
""", unsafe_allow_html=True)

# --- ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TWO_STAGE_DIR = os.path.join(PROCESSED_DIR, 'two_stage_results')
ASSETS_DIR = os.path.join(BASE_DIR, 'src', 'assets')

# --- BẢN ĐỒ TÊN CHỦ ĐỀ ---
TOPIC_NAMES = {
    1: "Du lịch & Hàng không", 4: "Giáo dục & Đào tạo", 5: "Bóng đá & Thể thao",
    12: "Bất động sản & Đầu tư", 26: "Chính trị & Công tác cán bộ", 7: "Thị trường Ô tô - Xe máy",
    23: "Thể thao & Giải đấu (Esports/Tennis)", 30: "Kinh tế & Phát triển Doanh nghiệp",
    25: "Thiên tai & Bão lũ", 21: "Pháp luật & An ninh trật tự", 14: "Y tế - Điều trị & Ung thư",
    10: "Đời sống dân sinh & Xã hội", 19: "Chiến sự Nga - Ukraine & Quốc tế", 11: "Văn học & Sách",
    6: "Công nghệ (Smartphone / Thiết bị số)", 2: "Giải trí & Điện ảnh / Hoa hậu",
    22: "Tài chính - Ngân hàng / Lãi suất", 16: "Thị trường Vàng & Tài chính",
    3: "Chính trị Quốc tế (Mỹ - Trung)", 13: "Kiến trúc & Không gian sống",
    32: "Y tế - Dinh dưỡng & Sức khỏe", 33: "Thời trang & Phong cách",
    15: "Tài chính Doanh nghiệp (Lợi nhuận)", 24: "Văn hóa & Nghệ thuật",
    27: "Thi cử & Tuyển sinh", 8: "Chính sách Y tế & Bệnh viện",
    31: "Thị trường Chứng khoán", 20: "Giao thông & An toàn đường bộ",
    18: "Đất đai & Pháp lý BĐS", 17: "Ẩm thực & Địa điểm ăn uống",
    9: "Vụ án & Tội phạm (Ma túy/Lừa đảo)", 29: "Bóng đá Quốc tế & FIFA",
    0: "Pháp luật (Xét xử/Tham nhũng)", 28: "Đời sống & Xã hội (Chuyện lạ)",
}

# --- 1. HÀM NẠP DỮ LIỆU & MODEL ---
@st.cache_resource
def load_data_and_models():
    try:
        vec = pickle.load(open(os.path.join(PROCESSED_DIR, 'lsa_tfidf_vectorizer.pkl'), 'rb'))
        lsa = pickle.load(open(os.path.join(PROCESSED_DIR, 'lsa_model.pkl'), 'rb'))
        km = pickle.load(open(os.path.join(TWO_STAGE_DIR, 'kmeans_two_stage.pkl'), 'rb'))
        
        stopwords_path = os.path.join(ASSETS_DIR, 'stopwords', 'vietnamese-stopwords.txt')
        sw = set()
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                sw = set(line.strip() for line in f)

        df_path = os.path.join(TWO_STAGE_DIR, 'two_stage_clusters.csv')
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            df = df.dropna(subset=['processed_content'])
            df = df[df['processed_content'].str.strip() != '']
            df.reset_index(drop=True, inplace=True)
            
            corpus = df['processed_content'].tolist()
            vec_tfidf = vec.transform(corpus)
            matrix = lsa.transform(vec_tfidf)
        else:
            df = None
            matrix = None

        return vec, lsa, km, sw, df, matrix
    except Exception as e:
        st.error(f"Lỗi nạp dữ liệu: {e}")
        return None, None, None, None, None, None

vectorizer, lsa_model, kmeans, stopwords, df_data, lsa_matrix = load_data_and_models()

# --- 2. HÀM TÍNH TOÁN TOP TRENDING ---
@st.cache_data
def get_trending_topics(_df, _matrix, _kmeans, _vectorizer, _lsa_model):
    if _df is None: return []
    
    distances = _kmeans.transform(_matrix)
    terms = _vectorizer.get_feature_names_out()
    centroids_tfidf = _lsa_model.named_steps['truncatedsvd'].inverse_transform(_kmeans.cluster_centers_)
    ordered_centroids = centroids_tfidf.argsort()[:, ::-1]
    cluster_counts = _df['cluster'].value_counts().sort_values(ascending=False)
    
    trending_list = []
    
    for cluster_id, count in cluster_counts.head(10).items():
        # Lấy từ khóa
        top_k_idx = ordered_centroids[cluster_id, :8]
        keywords = [terms[i] for i in top_k_idx]
        
        # Tìm bài báo đại diện
        indices = _df.index[_df['cluster'] == cluster_id].tolist()
        dists = distances[indices, cluster_id]
        sorted_idx = np.argsort(dists)
        
        rep_title = "Đang cập nhật..."
        rep_link = "#"
        related_articles = [] # Danh sách 10 bài báo liên quan
        
        valid_count = 0
        # Lấy top 30 bài gần nhất để chọn lọc
        for idx in sorted_idx[:40]: 
            real_idx = indices[idx]
            row = _df.loc[real_idx]
            title = str(row['title']).strip()
            
            if title.lower() != 'không tìm thấy' and len(title) > 10:
                if valid_count == 0:
                    # Bài đầu tiên là bài đại diện chính
                    rep_title = title
                    rep_link = row['url']
                else:
                    # Các bài tiếp theo thêm vào danh sách liên quan
                    if len(related_articles) < 10:
                        related_articles.append({'title': title, 'url': row['url']})
                
                valid_count += 1
                if len(related_articles) >= 10: # Đủ 10 bài thì dừng
                    break
        
        trending_list.append({
            'id': cluster_id,
            'count': count,
            'rep_title': rep_title,
            'rep_link': rep_link,
            'keywords': ", ".join(keywords),
            'related': related_articles
        })
        
    return trending_list

# --- 3. GIAO DIỆN CHÍNH ---
# Header
st.markdown("<h1 style='text-align: center; color: #d32f2f;'>🔥 CÁC CHỦ ĐỀ TIN TỨC NỔI BẬT NHẤT 🔥</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-size: 18px; color: #555;'>Hệ thống tự động tổng hợp và phân tích từ <b>{len(df_data)}</b> bài báo mới nhất</p>", unsafe_allow_html=True)
st.markdown("---")

# Hiển thị danh sách
col_spacer1, col_content, col_spacer2 = st.columns([1, 6, 1])

with col_content:
    if df_data is not None:
        with st.spinner("Đang phân tích dữ liệu nóng..."):
            trending_topics = get_trending_topics(df_data, lsa_matrix, kmeans, vectorizer, lsa_model)
        
        for i, topic in enumerate(trending_topics):
            topic_real_name = TOPIC_NAMES.get(topic['id'], f"Chủ đề {topic['id']}")
            
            # HIỂN THỊ THẺ CHỦ ĐỀ
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
            
            # PHẦN MỞ RỘNG: 10 BÀI BÁO LIÊN QUAN
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
            
            st.markdown("</div>", unsafe_allow_html=True) # Đóng thẻ top-card
            st.write("") 
            
    else:
        st.error("Không tìm thấy dữ liệu. Vui lòng kiểm tra lại.")

# Footer
st.markdown("---")
st.caption("Đồ án Tốt nghiệp | Sinh viên: Vũ Huy Hưng | GVHD: TS. Nguyễn Mạnh Hiển")