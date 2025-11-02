import pandas as pd
import glob
import os
import re
import sys
from underthesea import word_tokenize

# ---GỘP DỮ LIỆU TỪ CÁC FILE CSV ---
print("Bắt đầu gộp dữ liệu từ các file CSV thô...")
data_path = 'data/raw'
all_csv_files = glob.glob(os.path.join(data_path, "*.csv"))

if not all_csv_files:
    print(f"Lỗi: Không tìm thấy file CSV nào trong thư mục '{data_path}'. Hãy chạy scraper.py trước.")
    sys.exit(1)

list_of_dfs = []
for file in all_csv_files:
    try:
        df_temp = pd.read_csv(file)
        if all(col in df_temp.columns for col in ['url', 'title', 'content']):
             list_of_dfs.append(df_temp[['url', 'title', 'content']])
             print(f"Đã đọc file: {os.path.basename(file)} - có {len(df_temp)} bài báo.")
        else:
             print(f"Cảnh báo: File {os.path.basename(file)} thiếu cột. Bỏ qua.")
    except Exception as e:
        print(f"Lỗi khi đọc file {os.path.basename(file)}: {e}. Bỏ qua.")

if not list_of_dfs:
    print("Lỗi: Không đọc được dữ liệu hợp lệ.")
    sys.exit(1)

combined_df = pd.concat(list_of_dfs, ignore_index=True)
print(f"\nGộp dữ liệu thành công! Tổng số bài báo: {len(combined_df)}")
print("Kiểm tra dữ liệu thiếu (NaN) trước xử lý:")
print(combined_df.isnull().sum())
print("-" * 50)

# --- TIỀN XỬ LÝ VĂN BẢN ---
stopwords_path = 'src/preprocessing/assets/vietnamese-stopwords.txt'
try:
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        # Đọc và loại bỏ khoảng trắng thừa từ mỗi dòng, tạo thành set
        stopwords = set(line.strip() for line in f if line.strip())
    print(f"Đã tải {len(stopwords)} từ dừng từ file: {stopwords_path}")
except FileNotFoundError:
    print(f"Cảnh báo: Không tìm thấy file stopwords tại '{stopwords_path}'.")
    stopwords = set()

# --- HÀM PREPROCESS_TEXT ---
def preprocess_text(text):
    """Làm sạch văn bản thô: xóa link/email, chuyển chữ thường, xóa ký tự đặc biệt, tách từ, bỏ từ dừng."""
    # 1. Xử lý giá trị NaN
    if pd.isna(text):
        return ""
    text = str(text)
    # Xóa http/https/ftp links
    text = re.sub(r'https?://[^\s\n\r]+', '', text)
    # Xóa www links
    text = re.sub(r'www\.[^\s\n\r]+', '', text)
    # Xóa các tên miền .com/.vn/.net/... 
    text = re.sub(r'(?<!\w)(\S+\.(?:com|vn|net|org|edu|gov|info|biz|asia|mobi|name|pro|tel|xxx|xyz|me|tv|cc|ws|io|ai|gl|us|uk|ca|au|de|fr|jp|cn|ru|br|in|eu))(?!\w)', '', text)
    # Xóa email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # 3. Chuyển thành chữ thường
    text = text.lower()
    # 4. Loại bỏ ký tự đặc biệt, số (giữ lại chữ TV và dấu cách)
    # Thay thế bằng dấu cách để tránh dính chữ
    text = re.sub(r'[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]', ' ', text)
    # 5. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    # 6. Tách từ
    try:
        # Sử dụng format="text" để trả về chuỗi, sau đó split()
        tokens = word_tokenize(text, format="text").split()
    except Exception as e: # Bắt lỗi cụ thể hơn nếu cần
        # print(f"Lỗi khi tách từ: {e} - Văn bản gốc: {text[:100]}...") # Bỏ comment nếu muốn debug
        tokens = text.split() # Tách tạm bằng khoảng trắng nếu underthesea lỗi

    # 7. Loại bỏ từ dừng và từ quá ngắn
    # Giả định 'stopwords' là một set các từ dừng đã được tải
    processed_tokens = [word for word in tokens if word not in stopwords and len(word) > 1]

    # 8. Ghép các từ lại thành một chuỗi
    return ' '.join(processed_tokens)

print("\nBắt đầu tiền xử lý văn bản...")
combined_df['processed_content'] = combined_df['content'].apply(preprocess_text)
print("Tiền xử lý hoàn tất!")

print("\nKiểm tra dữ liệu thiếu (NaN) sau xử lý:")
print(combined_df['processed_content'].isnull().sum())

initial_rows = len(combined_df)
# Xóa các dòng có processed_content là NaN hoặc chuỗi rỗng
combined_df.dropna(subset=['processed_content'], inplace=True)
# Kiểm tra chuỗi rỗng sau khi loại bỏ khoảng trắng thừa
combined_df = combined_df[combined_df['processed_content'].str.strip() != '']
final_rows = len(combined_df)
print(f"Đã xóa {initial_rows - final_rows} dòng có nội dung rỗng sau tiền xử lý.")
print("-" * 50)


# --- PHẦN 3: LƯU KẾT QUẢ ---
output_folder = 'data/processed'
os.makedirs(output_folder, exist_ok=True)
processed_data_path = os.path.join(output_folder, 'processed_data.csv')

try:
    # Chỉ lưu các cột cần thiết
    combined_df[['url', 'title', 'processed_content']].to_csv(processed_data_path, index=False, encoding='utf-8-sig')
    print(f"\nĐã lưu dữ liệu đã xử lý vào file: '{processed_data_path}'")
    print("\nXem thử 5 dòng đầu tiên:")
    # Hiển thị nhiều nội dung hơn để kiểm tra
    pd.set_option('display.max_colwidth', 100)
    print(combined_df[['title', 'processed_content']].head())
except Exception as e:
    print(f"\nLỗi khi lưu file CSV: {e}")