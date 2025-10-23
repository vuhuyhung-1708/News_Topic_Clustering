import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import sys
from urllib.parse import urlparse
from config import SITES_CONFIG

# Headers đầy đủ
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Connection': 'keep-alive'
}

# --- Các hàm get_full_url, get_article_links, scrape_article_content giữ nguyên ---
def get_full_url(base_url, href):
    if href.startswith('http'):
        return href
    return base_url.rstrip('/') + '/' + href.lstrip('/')

def get_article_links(site_config, category_url):
    print(f"\nĐang lấy link từ trang: {site_config['name']} - {category_url}")
    try:
        response = requests.get(category_url, headers=HEADERS, timeout=15) # Tăng timeout
        response.raise_for_status() # Kiểm tra lỗi HTTP
        soup = BeautifulSoup(response.content, 'html.parser')
        all_links_found = []
        for selector in site_config['link_selector']:
            link_tags = soup.select(selector)
            if link_tags:
                print(f"  Áp dụng selector thành công: '{selector}'")
                links = [get_full_url(site_config.get('base_url', ''), tag['href']) for tag in link_tags if tag and 'href' in tag.attrs]
                all_links_found.extend(links)
        if all_links_found:
            unique_links = list(dict.fromkeys(all_links_found))
            print(f"  Tìm thấy {len(unique_links)} link bài báo.")
            return unique_links
        print("  Không tìm thấy link bài báo nào với các selector đã cấu hình.")
        return []
    except requests.exceptions.Timeout:
        print(f"  Lỗi: Hết thời gian chờ khi kết nối tới {category_url}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  Lỗi mạng khi lấy link: {e}")
        return []
    except Exception as e:
        print(f"  Lỗi không xác định khi lấy link: {e}")
        return []

def scrape_article_content(url, selectors):
    print(f"  Đang thu thập nội dung từ: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15) # Tăng timeout
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.select_one(selectors['title'])
        title = title_tag.get_text(strip=True) if title_tag else "Không tìm thấy"
        desc_tag = soup.select_one(selectors['description'])
        description = desc_tag.get_text(strip=True) if desc_tag else ""
        content_tags = soup.select(selectors['content'])
        content = '\n'.join([p.get_text(strip=True) for p in content_tags if p])
        if not content:
            print(f"  -> Cảnh báo: Bỏ qua bài báo '{url}' vì không có nội dung chính.")
            return None
        return {'url': url, 'title': title, 'description': description, 'content': content}
    except requests.exceptions.Timeout:
         print(f"  -> Lỗi: Hết thời gian chờ khi kết nối tới bài báo {url}")
         return None
    except requests.exceptions.RequestException as e:
        print(f"  -> Lỗi mạng khi thu thập bài báo {url}: {e}")
        return None
    except Exception as e:
        print(f"  -> Lỗi không xác định khi thu thập bài báo {url}: {e}")
        return None

# --- PHẦN THỰC THI CHÍNH ĐÃ THAY ĐỔI ---
if __name__ == '__main__':
    config = SITES_CONFIG
    print("\n" + "="*50)
    print("CHƯƠNG TRÌNH THU THẬP DỮ LIỆU TỪ NHIỀU LINK")
    print("="*50)

    # 1. Nhận danh sách link từ người dùng
    links_input = input("Vui lòng điền các link chuyên mục muốn thu thập, cách nhau bằng dấu phẩy (,):\n")
    category_urls = [link.strip() for link in links_input.split(',') if link.strip()] # Tách link và loại bỏ khoảng trắng thừa

    if not category_urls:
        print("Lỗi: Bạn chưa nhập link nào.")
        sys.exit(1)

    print(f"\nBắt đầu xử lý {len(category_urls)} link...")
    print("-" * 50)

    # 2. Vòng lặp xử lý từng link
    for i, category_url in enumerate(category_urls):
        print(f"\nĐang xử lý link {i+1}/{len(category_urls)}: {category_url}")
        
        # Thêm try...except để bỏ qua link lỗi và tiếp tục
        try:
            # 3. Tự động nhận diện trang báo
            domain = urlparse(category_url).netloc
            site_key = None
            for key in config:
                if key in domain:
                    site_key = key
                    break
            
            if site_key is None:
                print("  Lỗi: Trang web này không được hỗ trợ. Bỏ qua link này.")
                continue # Chuyển sang link tiếp theo

            print(f"  Nhận diện trang báo: {config[site_key]['name']}")
            site_config = config[site_key]

            # 4. Lấy danh sách link bài báo
            article_links = get_article_links(site_config, category_url)
            
            if not article_links:
                print("  Không có link bài báo nào được tìm thấy. Chuyển sang link tiếp theo.")
                continue # Chuyển sang link tiếp theo

            links_to_scrape = article_links[:30]
            print(f"  Sẽ thu thập {len(links_to_scrape)} bài báo mới nhất...")

            # 5. Thu thập nội dung chi tiết
            all_articles_data = []
            for link in links_to_scrape:
                data = scrape_article_content(link, site_config['article_selectors'])
                if data:
                    all_articles_data.append(data)
                time.sleep(1) # Giữ khoảng nghỉ

            # 6. Lưu kết quả ra file riêng cho link này
            if all_articles_data:
                df = pd.DataFrame(all_articles_data)
                os.makedirs('data', exist_ok=True)
                
                # Tạo tên file động
                category_name = urlparse(category_url).path.strip('/').replace('/', '_') or "trang-chu"
                if category_name.endswith(".htm") or category_name.endswith(".html"):
                    category_name = category_name.split('.')[0]
                output_filename = f"{site_key}_{category_name}.csv"
                full_path = os.path.join('data', output_filename)
                
                # Ghi đè file cũ nếu đã tồn tại
                df.to_csv(full_path, index=False, encoding='utf-8-sig')
                
                print(f"  Hoàn thành! Đã thu thập và lưu thành công {len(all_articles_data)} bài báo.")
                print(f"  File đã được lưu tại: {full_path}")
            else:
                print("  Cảnh báo: Không thu thập được dữ liệu từ bất kỳ bài báo nào cho link này.")

        except Exception as e:
            print(f"  Lỗi nghiêm trọng xảy ra khi xử lý link {category_url}: {e}. Bỏ qua link này.")
        
        print("-" * 50) # In dấu ngăn cách giữa các link

    print("\nĐã xử lý xong tất cả các link.")