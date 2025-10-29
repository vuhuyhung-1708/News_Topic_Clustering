# file: project/config.py

# Dictionary này chứa toàn bộ cấu hình cho các trang báo (Đã được cập nhật)
SITES_CONFIG = {
  "vnexpress": { # ok
    "name": "VnExpress",
    "link_selector": ["h3.title-news a"],
    "article_selectors": { "title": "h1.title-detail", "description": "p.description", "content": "article.fck_detail p.Normal" }
  },
  "znews": { # ok
    "name": "ZNews",
    "link_selector": ["article.article-item .article-title a"],
    "article_selectors": { "title": "h1.the-article-title", "description": "p.the-article-summary", "content": "div.the-article-body p" },
    "base_url": "https://znews.vn"
  },
  "thanhnien": { # ok
    "name": "Thanh Niên",
    "link_selector": ["a.story__title", "a.box-category-link-title"],
    "article_selectors": { "title": "h1.detail-title", "description": "h2.detail-sapo", "content": "div.detail-cmain p" },
    "base_url": "https://thanhnien.vn"
  },
  "tuoitre": { # ok
    "name": "Tuổi Trẻ",
    "link_selector": ["h3.box-title-text a"],
    "article_selectors": { "title": "h1.detail-title", "description": "h2.detail-sapo", "content": "div.detail-cmain p" },
    "base_url": "https://tuoitre.vn"
  },
  "dantri": { # ok
    "name": "Dân trí",
    "link_selector": ["h3.article-title a", "h2.article-title a"],
    "article_selectors": { "title": "h1.title-page", "description": "h2.singular-sapo", "content": "div.singular-content p" },
    "base_url": "https://dantri.com.vn"
  },
  "tienphong": { #ok
    "name": "Tiền Phong",
    "link_selector": ["article.story h2 a"],
    "article_selectors": { "title": "h1.article__title", "description": "div.article__sapo", "content": "div.article__body p" },
    "base_url": "https://tienphong.vn"
  },
  "vietnamnet": { # ok
    "name": "VietnamNet",
    "link_selector": ["h3.vnn-title a", "h4.vnn-title a"],
    "article_selectors": { "title": "h1.content-title", "description": "div.content-detail-sapo h2", "content": "div.maincontent-detail p, div.main-content-body p, div#maincontent p" },
    "base_url": "https://vietnamnet.vn"
  },
  "baochinhphu": { # ok
    "name": "Báo Chính Phủ (VGP News)",
    "link_selector": ["h2.box-title-text a", "h3.box-title a"],
    "article_selectors": { "title": "h1.detail__title", "description": "div.detail__summary", "content": "div.detail__content p" },
    "base_url": "https://baochinhphu.vn"
  },
  "vov": { # ok
    "name": "VOV (Đài Tiếng nói Việt Nam)",
    "link_selector": ['h3[class^="title-"] a'],
    "article_selectors": { "title": "h1.main-title", "description": "h2.sapo", "content": "div.content-detail p" },
    "base_url": "https://vov.vn"
  },

  # --- 5 TRANG BÁO MỚI ĐƯỢC BỔ SUNG ---
  "cand": {
      "name": "Công an Nhân dân",
      "link_selector": ["div.box-title h3 a", "div.box-sub-title h4 a"],
      "article_selectors": {
          "title": "h1.news-title",
          "description": "div.news-description",
          "content": "div.news-content p"
       },
       "base_url": "https://cand.com.vn"
  },
  "qdnd": {
      "name": "Quân đội Nhân dân",
      "link_selector": ["h3.post-title a", "h4.post-title a"],
      "article_selectors": {
          "title": "h1.post-title",
          "description": "p.post-description",
          "content": "div.post-content p"
      },
      "base_url": "https://www.qdnd.vn"
  },
  "sggp": {
      "name": "Sài Gòn Giải Phóng",
      "link_selector": ["div.title_news h3 a"],
      "article_selectors": {
          "title": "h1.title_detail",
          "description": "div.sapo_detail",
          "content": "div.content_detail p"
      },
      "base_url": "https://www.sggp.org.vn"
  },
  "hanoimoi": {
      "name": "Hà Nội Mới",
      "link_selector": ["h4.media-heading a"],
      "article_selectors": {
          "title": "h1.entry-title",
          "description": "div.entry-description",
          "content": "div.entry-content p"
      },
      "base_url": "https://hanoimoi.vn"
  },
  "baotintuc": {
      "name": "Báo Tin tức (TTXVN)",
      "link_selector": ["h3.title-news a"],
      "article_selectors": {
          "title": "h1.detail-title",
          "description": "div.detail-sapo",
          "content": "div.detail-content p"
       },
       "base_url": "https://baotintuc.vn"
  }
}