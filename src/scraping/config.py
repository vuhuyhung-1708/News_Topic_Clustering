SITES_CONFIG = {
   
    "vnexpress": {
        "name": "VnExpress",
        "link_selector": ["h3.title-news a", "h2.title-news a", "article.item-news h3 a"],
        "article_selectors": { "title": "h1.title-detail", "description": "p.description", "content": "article.fck_detail p.Normal" }
    },
    "znews": {
        "name": "ZNews",
        "link_selector": ["article.article-item .article-title a", "p.article-title a"],
        "article_selectors": { "title": "h1.the-article-title", "description": "p.the-article-summary", "content": "div.the-article-body p" },
        "base_url": "https://znews.vn"
    },
    "thanhnien": {
        "name": "Thanh Niên",
        "link_selector": ["a.story__title", "a.box-category-link-title", "div.zone--timeline article h2 a"],
        "article_selectors": { "title": "h1.detail-title", "description": "h2.detail-sapo", "content": "div.detail-cmain p" },
        "base_url": "https://thanhnien.vn"
    },
    "tuoitre": {
        "name": "Tuổi Trẻ",
        "link_selector": ["h3.box-title-text a", "h3.title-news a"],
        "article_selectors": { "title": "h1.detail-title", "description": "h2.detail-sapo", "content": "div.detail-cmain p" },
        "base_url": "https://tuoitre.vn"
    },
    "dantri": {
        "name": "Dân trí",
        "link_selector": ["h3.article-title a", "h2.article-title a"],
        "article_selectors": { "title": "h1.title-page", "description": "h2.singular-sapo", "content": "div.singular-content p" },
        "base_url": "https://dantri.com.vn"
    },
    "tienphong": {
        "name": "Tiền Phong",
        "link_selector": ["article.story h2 a", "div.story__heading a"],
        "article_selectors": { "title": "h1.article__title", "description": "div.article__sapo", "content": "div.article__body p" },
        "base_url": "https://tienphong.vn"
    },
    "vietnamnet": {
        "name": "VietnamNet",
        "link_selector": ["h3.vnn-title a", "h4.vnn-title a", "div.feature-box__content h3 a"],
        "article_selectors": { "title": "h1.content-title", "description": "div.content-detail-sapo h2", "content": "div.maincontent-detail p, div.main-content-body p, div#maincontent p" },
        "base_url": "https://vietnamnet.vn"
    },
    "soha": { 
        "name": "Soha",
        "link_selector": ["h3.box-detail-title a", "h3.news-title a", "h3 a"],
        "article_selectors": { 
            "title": "h1.news-title", 
            "description": "h2.news-sapo", 
            "content": "div.news-content p, div.detail-content p" 
        },
        "base_url": "https://soha.vn"
    },
    "cafef": { 
        "name": "CafeF",
        "link_selector": ["h3 a", "h4 a", "div.knswli h3 a"],
        "article_selectors": { 
            "title": "h1.title", 
            "description": "h2.sapo", 
            "content": "div.detail-content p, div.contentdetail p" 
        },
        "base_url": "https://cafef.vn"
    },
    "genk": { 
        "name": "GenK",
        "link_selector": ["h4.knswli-title a", "h3.knswli-title a", "li.knswli h4 a"],
        "article_selectors": { 
            "title": "h1.kbwc-title", 
            "description": "h2.knc-sapo", 
            "content": "div.knc-content p" 
        },
        "base_url": "https://genk.vn"
    },
    "kenh14": { 
        "name": "Kênh 14",
        "link_selector": ["h3.knswli-title a", "h4.knswli-title a", "li.knswli h3 a"],
        "article_selectors": { 
            "title": "h1.kbwc-title, h1.title", 
            "description": "h2.knc-sapo, h2.sapo", 
            "content": "div.knc-content p, div.detail-content p, div.content p" 
        },
        "base_url": "https://kenh14.vn"
    },
    "danviet": { 
        "name": "Dân Việt",
        "link_selector": ["div.box7-list-news h3 a", "h3 a", "h4 a"],
        "article_selectors": { 
            "title": "h1.title-page-detail", 
            "description": "div.sapo", 
            "content": "div.entry-body p" 
        },
        "base_url": "https://danviet.vn"
    },
  

}