# file: project/config.py

SITES_CONFIG = {
  "vnexpress": { # ok
    "name": "VnExpress",
    "link_selector": ["h3.title-news a"],
    "article_selectors": {
      "title": "h1.title-detail",
      "description": "p.description",
      "content": "article.fck_detail p.Normal"
    }
  },
  "znews": { # ok
    "name": "ZNews",
    "link_selector": ["article.article-item .article-title a"],
    "article_selectors": { "title": "h1.the-article-title", "description": "p.the-article-summary", "content": "div.the-article-body p" },
    "base_url": "https://znews.vn"
  },
  "thanhnien": { # ok
    "name": "Thanh Niên",
    "link_selector": [
        "a.story__title",
        "a.box-category-link-title"
    ],
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
    "link_selector": [
        "h3.article-title a",
        "h2.article-title a" 
    ],
    "article_selectors": {
      "title": "h1.title-page",
      "description": "h2.singular-sapo",
      "content": "div.singular-content p"
    },
    "base_url": "https://dantri.com.vn"
  },
  "vietnamnet": { # ok
    "name": "VietnamNet",
    "link_selector": [
        "h3.vnn-title a",
        "h4.vnn-title a"
    ],
    "article_selectors": {
      "title": "h1.content-title",
      "description": "div.content-detail-sapo h2",
      "content": "div.maincontent-detail p, div.main-content-body p, div#maincontent p"
    },
    "base_url": "https://vietnamnet.vn"
  },
  "tienphong": { #ok
    "name": "Tiền Phong",
    "link_selector": ["article.story h2 a"],
     "article_selectors": {
      "title": "h1.article__title",
      "description": "div.article__sapo",
      "content": "div.article__body p"
    },
    "base_url": "https://tienphong.vn"
  }  
}