import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import sys
from urllib.parse import urlparse, urljoin 
from concurrent.futures import ThreadPoolExecutor, as_completed
import random 

from config import SITES_CONFIG

# --- CẤU HÌNH MỞ RỘNG ---
MAX_PAGES_TO_SCAN = 10       # Số trang danh sách tối đa sẽ quét (VD: quét từ trang 1 -> 15)
MAX_ARTICLES_PER_CATEGORY = 200  # Tăng giới hạn số bài lấy về (thay vì 70)
MAX_WORKERS = 10             # Tăng số luồng để tải nhanh hơn

# Headers chung
HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Connection': 'keep-alive'
}

# DANH SÁCH USER-AGENTS ĐỂ XOAY VÒNG
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

def get_full_url(base_url, href):
    """Tạo URL đầy đủ một cách an toàn."""
    return urljoin(base_url, href)

def generate_paginated_url(base_url, page_num):
    """
    Tự động tạo URL phân trang. 
    Hầu hết báo VN dùng tham số ?page=X hoặc &page=X
    """
    if page_num == 1:
        return base_url
    
    separator = '&' if '?' in base_url else '?'
    return f"{base_url}{separator}page={page_num}"

def get_links_from_single_page(site_config, page_url):
    """Lấy danh sách link bài báo từ 1 trang cụ thể."""
    print(f"  -> Đang quét trang: {page_url} ...")
    try:
        request_headers = HEADERS.copy()
        request_headers['User-Agent'] = random.choice(USER_AGENTS)
        
        response = requests.get(page_url, headers=request_headers, timeout=10)
        if response.status_code == 404:
            print("     [404] Trang không tồn tại.")
            return []
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        page_links = []
        for selector in site_config['link_selector']:
            link_tags = soup.select(selector)
            if link_tags:
                links = [get_full_url(page_url, tag['href']) for tag in link_tags if tag and 'href' in tag.attrs]
                page_links.extend(links)
                
        # Lọc trùng ngay trên trang này
        return list(dict.fromkeys(page_links))
            
    except Exception as e:
        print(f"     Lỗi quét trang: {e}")
        return []

def scrape_article_content(url, selectors):
    """Thu thập nội dung chi tiết bài báo."""
    try:
        request_headers = HEADERS.copy()
        request_headers['User-Agent'] = random.choice(USER_AGENTS)
        
        response = requests.get(url, headers=request_headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title_tag = soup.select_one(selectors['title'])
        title = title_tag.get_text(strip=True) if title_tag else "Không tìm thấy"
        
        desc_tag = soup.select_one(selectors['description'])
        description = desc_tag.get_text(strip=True) if desc_tag else ""
        
        content_tags = soup.select(selectors['content'])
        content = '\n'.join([p.get_text(strip=True) for p in content_tags if p])
        
        # Bỏ qua bài quá ngắn (thường là video/ảnh)
        if not content or len(content) < 200:
            return None
            
        return {'url': url, 'title': title, 'description': description, 'content': content}
    except Exception as e:
        return None

if __name__ == '__main__':
    config = SITES_CONFIG
    print("\n" + "="*60)
    print("CHƯƠNG TRÌNH THU THẬP DỮ LIỆU (DEEP CRAWL - PHÂN TRANG)")
    print(f"Cấu hình: Quét tối đa {MAX_PAGES_TO_SCAN} trang/chuyên mục | Lấy tối đa {MAX_ARTICLES_PER_CATEGORY} bài")
    print("="*60)

    links_input = input("Nhập link chuyên mục (cách nhau dấu phẩy):\n")
    category_urls = [link.strip() for link in links_input.split(',') if link.strip()]

    if not category_urls:
        print("Lỗi: Bạn chưa nhập link nào.")
        sys.exit(1)

    # VÒNG LẶP QUA TỪNG CHUYÊN MỤC
    for i, category_url in enumerate(category_urls):
        print(f"\n🚀 Đang xử lý chuyên mục {i+1}/{len(category_urls)}: {category_url}")
        
        # 1. NHẬN DIỆN SITE
        domain = urlparse(category_url).netloc
        site_key = None
        for key in config:
            if key in domain:
                site_key = key
                break
        
        if site_key is None:
            print("  ❌ Lỗi: Trang web không được hỗ trợ trong config.")
            continue

        site_config = config[site_key]
        print(f"  Báo: {site_config['name']}")

        # 2. VÒNG LẶP PHÂN TRANG ĐỂ LẤY NHIỀU LINK HƠN
        all_potential_links = []
        print(f"  Bắt đầu quét danh sách link (Tối đa {MAX_PAGES_TO_SCAN} trang)...")
        
        for page_num in range(1, MAX_PAGES_TO_SCAN + 1):
            # Nếu đã đủ số lượng link cần thiết thì dừng quét trang
            if len(all_potential_links) >= MAX_ARTICLES_PER_CATEGORY + 50: # Lấy dư ra chút để bù trừ trùng lặp
                print(f"  -> Đã tìm thấy đủ số lượng link tiềm năng. Dừng quét trang.")
                break

            current_page_url = generate_paginated_url(category_url, page_num)
            links_on_page = get_links_from_single_page(site_config, current_page_url)
            
            if not links_on_page:
                print("  -> Không tìm thấy bài nào ở trang này. Dừng phân trang.")
                break
                
            prev_len = len(all_potential_links)
            # Thêm link mới vào danh sách (tránh trùng ngay lập tức)
            for lnk in links_on_page:
                if lnk not in all_potential_links:
                    all_potential_links.append(lnk)
            
            added = len(all_potential_links) - prev_len
            print(f"     + Tìm thấy {len(links_on_page)} link, Thêm mới: {added} (Tổng: {len(all_potential_links)})")
            
            if added == 0 and page_num > 1:
                print("  -> Trang này toàn link cũ. Dừng phân trang.")
                break
                
            time.sleep(1) # Nghỉ nhẹ giữa các trang

        # 3. LOẠI BỎ LINK ĐÃ CÓ TRONG CSV CŨ
        output_folder = 'data/raw'
        os.makedirs(output_folder, exist_ok=True)
        category_name = urlparse(category_url).path.strip('/').replace('/', '_') or "trang-chu"
        if category_name.endswith((".htm", ".html")): category_name = category_name.split('.')[0]
        output_filename = f"{site_key}_{category_name}.csv"
        full_path = os.path.join(output_folder, output_filename)

        existing_links = set()
        if os.path.exists(full_path):
            try:
                df_existing = pd.read_csv(full_path)
                if 'url' in df_existing.columns:
                    existing_links = set(df_existing['url'].tolist())
            except: pass
        
        # Lọc link chưa cào
        links_to_scrape = [lnk for lnk in all_potential_links if lnk not in existing_links]
        links_to_scrape = links_to_scrape[:MAX_ARTICLES_PER_CATEGORY] # Cắt đúng số lượng cần lấy
        
        print(f"\n  Tổng link tìm được: {len(all_potential_links)}")
        print(f"  Đã tồn tại trong file: {len(existing_links)}")
        print(f"  Sẽ cào mới: {len(links_to_scrape)} bài")

        if not links_to_scrape:
            print("  -> Không có bài mới để cào. Chuyển tiếp.")
            continue

        # 4. CÀO NỘI DUNG ĐA LUỒNG
        print(f"  Đang tải nội dung ({MAX_WORKERS} luồng)...")
        new_articles = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(scrape_article_content, url, site_config['article_selectors']): url 
                for url in links_to_scrape
            }
            
            completed_count = 0
            for future in as_completed(future_to_url):
                completed_count += 1
                try:
                    data = future.result()
                    if data:
                        new_articles.append(data)
                    print(f"    Tiến độ: {completed_count}/{len(links_to_scrape)} - Thành công: {len(new_articles)}...", end='\r')
                except: pass

        print(f"\n  Hoàn tất. Thu được {len(new_articles)} bài.")

        # 5. LƯU FILE
        if new_articles:
            df = pd.DataFrame(new_articles)
            if os.path.exists(full_path):
                df.to_csv(full_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                print(f"  ✅ Đã NỐI thêm vào: {full_path}")
            else:
                df.to_csv(full_path, mode='w', header=True, index=False, encoding='utf-8-sig')
                print(f"  ✅ Đã TẠO MỚI file: {full_path}")
        else:
            print("  ⚠️ Không thu được nội dung bài nào.")
            
        print("-" * 60)

    print("\nĐã xử lý xong tất cả!")