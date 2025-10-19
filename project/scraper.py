import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import sys
from urllib.parse import urlparse
from config import SITES_CONFIG

# Headers đầy đủ để giả lập trình duyệt, tránh bị chặn
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Connection': 'keep-alive'
}

def get_full_url(base_url, href):
    if href.startswith('http'):
        return href
    return base_url.rstrip('/') + '/' + href.lstrip('/')

def get_article_links(site_config, category_url):
    print(f"Đang lấy link từ trang: {site_config['name']} - {category_url}")
    try:
        response = requests.get(category_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        all_links_found = []
        for selector in site_config['link_selector']:
            link_tags = soup.select(selector)
            if link_tags:
                print(f"Áp dụng selector thành công: '{selector}'")
                links = [get_full_url(site_config.get('base_url', ''), tag['href']) for tag in link_tags if tag and 'href' in tag.attrs]
                all_links_found.extend(links)
        
        if all_links_found:
            unique_links = list(dict.fromkeys(all_links_found))
            print(f"Tìm thấy {len(unique_links)} link bài báo.")
            return unique_links
        print("Không tìm thấy link bài báo nào với các selector đã cấu hình.")
        return []
    except Exception as e:
        print(f"Lỗi khi lấy link: {e}")
        return []

def scrape_article_content(url, selectors):
    print(f"Đang thu thập nội dung từ: {url}") # Đã sửa
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title_tag = soup.select_one(selectors['title'])
        title = title_tag.get_text(strip=True) if title_tag else "Không tìm thấy"
        
        desc_tag = soup.select_one(selectors['description'])
        description = desc_tag.get_text(strip=True) if desc_tag else ""
        
        content_tags = soup.select(selectors['content'])
        content = '\n'.join([p.get_text(strip=True) for p in content_tags if p])

        if not content: return None
        return {'url': url, 'title': title, 'description': description, 'content': content}
    except Exception as e:
        print(f"Lỗi khi thu thập bài báo {url}: {e}")

if __name__ == '__main__':
    config = SITES_CONFIG
    print("\n" + "="*50)
    print("CHƯƠNG TRÌNH THU THẬP DỮ LIỆU TIN TỨC") 
    
    category_url = input("Vui lòng điền link chuyên mục bạn muốn thu thập: ") 

    domain = urlparse(category_url).netloc
    site_key = None
    for key in config:
        if key in domain:
            site_key = key
            break
            
    if site_key is None:
        print("\nLỗi: Trang web này không được hỗ trợ.")
        sys.exit(1)
        
    print(f"Nhận diện trang báo: {config[site_key]['name']}")
    site_config = config[site_key]
    
    article_links = get_article_links(site_config, category_url)
    
    if not article_links:
        sys.exit(0)

    links_to_scrape = article_links[:30]
    print(f"Sẽ thu thập {len(links_to_scrape)} bài báo mới nhất...") 

    all_articles_data = []
    for link in links_to_scrape:
        data = scrape_article_content(link, site_config['article_selectors'])
        if data:
            all_articles_data.append(data)
        time.sleep(1)

    if all_articles_data:
        df = pd.DataFrame(all_articles_data)
        os.makedirs('data', exist_ok=True)
        category_name = category_url.split('/')[-1].split('.')[0]
        if not category_name:
             category_name = category_url.split('/')[-2]

        output_filename = f"{site_key}_{category_name}.csv"
        full_path = os.path.join('data', output_filename)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(f"Hoàn thành! Đã thu thập và lưu thành công {len(all_articles_data)} bài báo.") # Đã sửa
        print(f"File đã được lưu tại: {full_path}")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("Cảnh báo: Không thu thập được dữ liệu từ bất kỳ bài báo nào.")
        print("="*50)