import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import sys
from urllib.parse import urlparse
from config import SITES_CONFIG

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Connection': 'keep-alive'
}

def get_full_url(base_url, href):
    """Tạo URL đầy đủ từ link tương đối (nếu cần)."""
    if href.startswith('http'):
        return href
    return base_url.rstrip('/') + '/' + href.lstrip('/')

def get_article_links(site_config, category_url):
    """Lấy danh sách link bài báo từ trang chuyên mục."""
    print(f"\nĐang lấy link từ trang: {site_config['name']} - {category_url}")
    try:
        response = requests.get(category_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
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
    """Thu thập tiêu đề, mô tả, nội dung chi tiết từ link một bài báo."""
    print(f"  Đang thu thập nội dung từ: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
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

if __name__ == '__main__':
    """Hàm chính: Nhận link, xác định trang, gọi hàm thu thập và lưu kết quả."""
    config = SITES_CONFIG
    print("\n" + "="*50)
    print("CHƯƠNG TRÌNH THU THẬP DỮ LIỆU TỪ NHIỀU LINK")
    print("="*50)

    links_input = input("Vui lòng điền các link chuyên mục muốn thu thập, cách nhau bằng dấu phẩy (,):\n")
    category_urls = [link.strip() for link in links_input.split(',') if link.strip()]

    if not category_urls:
        print("Lỗi: Bạn chưa nhập link nào.")
        sys.exit(1)

    print(f"\nBắt đầu xử lý {len(category_urls)} link...")
    print("-" * 50)

    for i, category_url in enumerate(category_urls):
        print(f"\nĐang xử lý link {i+1}/{len(category_urls)}: {category_url}")
        try:
            domain = urlparse(category_url).netloc
            site_key = None
            for key in config:
                if key in domain:
                    site_key = key
                    break
            
            if site_key is None:
                print("  Lỗi: Trang web này không được hỗ trợ. Bỏ qua link này.")
                continue

            print(f"  Nhận diện trang báo: {config[site_key]['name']}")
            site_config = config[site_key]
            article_links = get_article_links(site_config, category_url)
            
            if not article_links:
                print("  Không có link bài báo nào được tìm thấy. Chuyển sang link tiếp theo.")
                continue

            links_to_scrape = article_links[:50]
            print(f"  Sẽ thu thập {len(links_to_scrape)} bài báo mới nhất...")

            all_articles_data = []
            for link in links_to_scrape:
                data = scrape_article_content(link, site_config['article_selectors'])
                if data:
                    all_articles_data.append(data)
                time.sleep(1)

            if all_articles_data:
                df = pd.DataFrame(all_articles_data)
                output_folder = 'data/raw'
                os.makedirs(output_folder, exist_ok=True)
                category_name = urlparse(category_url).path.strip('/').replace('/', '_') or "trang-chu"
                if category_name.endswith((".htm", ".html")):
                    category_name = category_name.split('.')[0]
                output_filename = f"{site_key}_{category_name}.csv"
                full_path = os.path.join(output_folder, output_filename)
                
                df.to_csv(full_path, index=False, encoding='utf-8-sig')
                
                print(f"  Hoàn thành! Đã thu thập và lưu thành công {len(all_articles_data)} bài báo.")
                print(f"  File đã được lưu tại: {full_path}")
            else:
                print("  Cảnh báo: Không thu thập được dữ liệu từ bất kỳ bài báo nào cho link này.")
        except Exception as e:
            print(f"  Lỗi nghiêm trọng xảy ra khi xử lý link {category_url}: {e}. Bỏ qua link này.")
        print("-" * 50)

    print("\nĐã xử lý xong tất cả các link.")

# import pandas as pd
# import os
# import time
# import sys
# from urllib.parse import urlparse

# # --- THAY ĐỔI QUAN TRỌNG: SỬ DỤNG CÁC THƯ VIỆN MỚI ---
# import requests
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.options import Options
# # Thư viện cho "chờ đợi thông minh"
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# # Thư viện xử lý lỗi Timeout của Selenium
# from selenium.common.exceptions import TimeoutException


# from config import SITES_CONFIG

# HEADERS = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
# }

# # --- HÀM THIẾT LẬP SELENIUM ---
# def setup_driver():
#     """Khởi tạo và cấu hình trình duyệt Chrome chạy ẩn."""
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")  # Chạy ở chế độ nền, không mở cửa sổ trình duyệt
#     chrome_options.add_argument("--log-level=3") # Tắt các log không cần thiết của Selenium
#     chrome_options.add_experimental_option('excludeSwitches', ['enable-logging']) # Tắt log của Chrome Driver
#     # Tự động cài đặt và quản lý ChromeDriver
#     service = ChromeService(ChromeDriverManager().install())
#     driver = webdriver.Chrome(service=service, options=chrome_options)
#     driver.set_page_load_timeout(30) # Đặt thời gian tối đa chờ tải trang
#     return driver

# def get_full_url(base_url, href):
#     """Tạo URL đầy đủ từ link tương đối (nếu cần)."""
#     if not href or href.startswith('javascript:'): # Bỏ qua các href không hợp lệ
#         return None
#     if href.startswith('http'):
#         return href
#     return base_url.rstrip('/') + '/' + href.lstrip('/')

# # --- HÀM NÀY ĐƯỢC NÂNG CẤP HOÀN TOÀN ---
# def get_article_links(site_config, category_url):
#     """Lấy danh sách link bài báo từ trang chuyên mục (sử dụng Selenium)."""
#     print(f"\nĐang lấy link từ trang: {site_config['name']} - {category_url} (sử dụng Selenium)")
#     driver = setup_driver()
#     all_links_found = []
#     try:
#         driver.get(category_url)

#         # Xác định selector để chờ (lấy selector đầu tiên từ config)
#         if not site_config['link_selector']:
#             print("  Lỗi: Chưa định nghĩa link_selector trong config.py")
#             return []
#         wait_selector = site_config['link_selector'][0]

#         # --- "CHỜ ĐỢI THÔNG MINH" ---
#         print(f"  Chờ tối đa 15 giây cho selector '{wait_selector}' xuất hiện...")
#         WebDriverWait(driver, 15).until(
#             EC.presence_of_all_elements_located((By.CSS_SELECTOR, wait_selector))
#         )
#         print("  Trang đã tải xong nội dung động.")

#         # Lấy mã HTML sau khi JavaScript đã chạy
#         soup = BeautifulSoup(driver.page_source, 'html.parser')

#         # Duyệt qua tất cả selector trong cấu hình
#         for selector in site_config['link_selector']:
#             link_tags = soup.select(selector)
#             if link_tags:
#                 print(f"  Áp dụng selector thành công: '{selector}'")
#                 links = [get_full_url(site_config.get('base_url', ''), tag.get('href')) # Sử dụng .get('href') an toàn hơn
#                          for tag in link_tags if tag]
#                 # Lọc bỏ các giá trị None trả về từ get_full_url
#                 all_links_found.extend([link for link in links if link])

#         if all_links_found:
#             unique_links = list(dict.fromkeys(all_links_found)) # Loại bỏ trùng lặp
#             print(f"  Tìm thấy {len(unique_links)} link bài báo.")
#             return unique_links

#         print("  Không tìm thấy link bài báo nào.")
#         return []

#     except TimeoutException:
#         print(f"  Lỗi: Hết thời gian chờ (15s) cho selector '{wait_selector}' xuất hiện. Trang tải quá chậm hoặc selector sai.")
#         return []
#     except Exception as e:
#         print(f"  Lỗi không xác định khi lấy link bằng Selenium: {e}")
#         return []
#     finally:
#         driver.quit() # Luôn đóng trình duyệt dù thành công hay thất bại

# # Hàm scrape_article_content giữ nguyên, dùng requests cho nhanh
# def scrape_article_content(url, selectors):
#     """Thu thập tiêu đề, mô tả, nội dung chi tiết từ link một bài báo (sử dụng Requests)."""
#     print(f"  Đang thu thập nội dung từ: {url}")
#     try:
#         response = requests.get(url, headers=HEADERS, timeout=15)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, 'html.parser')
#         title_tag = soup.select_one(selectors['title'])
#         title = title_tag.get_text(strip=True) if title_tag else "Không tìm thấy"
#         desc_tag = soup.select_one(selectors['description'])
#         description = desc_tag.get_text(strip=True) if desc_tag else ""
#         content_tags = soup.select(selectors['content'])
#         content = '\n'.join([p.get_text(strip=True) for p in content_tags if p])
#         if not content:
#             print(f"  -> Cảnh báo: Bỏ qua bài báo '{url}' vì không có nội dung chính.")
#             return None
#         return {'url': url, 'title': title, 'description': description, 'content': content}
#     except requests.exceptions.Timeout:
#          print(f"  -> Lỗi: Hết thời gian chờ khi kết nối tới bài báo {url}")
#          return None
#     except requests.exceptions.RequestException as e:
#         print(f"  -> Lỗi mạng khi thu thập bài báo {url}: {e}")
#         return None
#     except Exception as e:
#         print(f"  -> Lỗi không xác định khi thu thập bài báo {url}: {e}")
#         return None

# # Phần if __name__ == '__main__' giữ nguyên không đổi
# if __name__ == '__main__':
#     config = SITES_CONFIG
#     print("\n" + "="*50)
#     print("CHƯƠNG TRÌNH THU THẬP DỮ LIỆU TỪ NHIỀU LINK (NÂNG CẤP)")
#     print("="*50)

#     links_input = input("Vui lòng điền các link chuyên mục muốn thu thập, cách nhau bằng dấu phẩy (,):\n")
#     category_urls = [link.strip() for link in links_input.split(',') if link.strip()]

#     if not category_urls:
#         print("Lỗi: Bạn chưa nhập link nào.")
#         sys.exit(1)

#     print(f"\nBắt đầu xử lý {len(category_urls)} link...")
#     print("-" * 50)

#     for i, category_url in enumerate(category_urls):
#         print(f"\nĐang xử lý link {i+1}/{len(category_urls)}: {category_url}")
#         try:
#             domain = urlparse(category_url).netloc
#             site_key = None
#             for key in config:
#                 if key in domain:
#                     site_key = key
#                     break

#             if site_key is None:
#                 print("  Lỗi: Trang web này không được hỗ trợ trong config.py. Bỏ qua link này.")
#                 continue

#             print(f"  Nhận diện trang báo: {config[site_key]['name']}")
#             site_config = config[site_key]

#             # Gọi hàm lấy link bằng Selenium
#             article_links = get_article_links(site_config, category_url)

#             if not article_links:
#                 print("  Không có link bài báo nào được tìm thấy. Chuyển sang link tiếp theo.")
#                 continue

#             links_to_scrape = article_links[:30] # Giới hạn 30 bài
#             print(f"  Sẽ thu thập nội dung cho {len(links_to_scrape)} bài báo mới nhất...")

#             all_articles_data = []
#             for link in links_to_scrape:
#                 # Gọi hàm cào nội dung bằng Requests
#                 data = scrape_article_content(link, site_config['article_selectors'])
#                 if data:
#                     all_articles_data.append(data)
#                 time.sleep(1) # Giữ khoảng nghỉ giữa các request cào nội dung

#             if all_articles_data:
#                 df = pd.DataFrame(all_articles_data)
#                 output_folder = 'data/raw'
#                 os.makedirs(output_folder, exist_ok=True)
#                 category_name = urlparse(category_url).path.strip('/').replace('/', '_') or "trang-chu"
#                 if category_name.endswith((".htm", ".html")):
#                     category_name = category_name.split('.')[0]
#                 output_filename = f"{site_key}_{category_name}.csv"
#                 full_path = os.path.join(output_folder, output_filename)

#                 df.to_csv(full_path, index=False, encoding='utf-8-sig')

#                 print(f"  Hoàn thành! Đã thu thập và lưu thành công {len(all_articles_data)} bài báo.")
#                 print(f"  File đã được lưu tại: {full_path}")
#             else:
#                 print("  Cảnh báo: Không thu thập được dữ liệu nội dung từ bất kỳ bài báo nào cho link này.")

#         except Exception as e:
#             print(f"  Lỗi nghiêm trọng xảy ra khi xử lý link {category_url}: {e}. Bỏ qua link này.")

#         print("-" * 50)

#     print("\nĐã xử lý xong tất cả các link.")