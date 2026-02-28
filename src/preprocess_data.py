
import pandas as pd
import re
from underthesea import word_tokenize

class VietnameseTextPreprocessor:
    def __init__(self, csv_path, stopwords_path="../stopWords/vietnamese-stopwords-dash.txt"):
        """
        Khởi tạo bộ tiền xử lý dữ liệu báo chí tiếng Việt.
        """
        print(f"Lấy data từ: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        self.stopwords = set()
        if stopwords_path:
            print(f"Đang tải danh sách stopWords từ: {stopwords_path}")
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.stopwords.add(word)
            print(f"Tải thành công {len(self.stopwords)} stopWords.")
        else:
            print("Không sử dụng stopWords (stopwords_path=None)")

    def clean_text(self, text):
        """
        Làm sạch văn bản chuyên dụng cho Tiêu đề báo chí tiếng Việt
        """
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        # xoa cac ký tự không mong muốn như \xa0 (non-breaking space), &amp; (HTML entity)
        text = text.replace('\xa0', ' ').replace('&amp;', ' ')
        # xoa các ký tự đặc biệt, chỉ giữ lại chữ cái, số, dấu cách và một số ký tự quan trọng (% cho các bài viết về kinh tế)
        text = re.sub(r'[^\w\s%]', ' ', text)
        # Xóa các số đơn lẻ
        text = re.sub(r'\b\d+\b', ' ', text)
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def segment_words(self, text):
        """
        Tách từ tiếng Việt bằng thư viện underthesea và gán dấu gạch dưới (_)
        """
        if not text: return ""
        
        return word_tokenize(text, format="text")

    def remove_stopwords(self, text):
        """
        Loại bỏ các từ dừng (stopwords)
        """
        if not text: return ""
        words = text.split()
        
        filtered_words = [word for word in words if word not in self.stopwords]
        return " ".join(filtered_words)

    def process_pipeline(self, text_column='title', output_column='title_clean'):
        """
        Chạy toàn bộ quy trình tiền xử lý cho một cột cụ thể.
        """
        print(f"Bắt đâu tiền xử lý cột '{text_column}'...")
        
        # xoa cac dong khong co data (NaN) truoc khi tien hanh xu ly
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=[text_column])
        print(f"Đã xóa {initial_len - len(self.df)} dòng rỗng.")
            
        print("Loại bỏ các bài báo trùng lặp...")
        self.df = self.df.drop_duplicates(subset=['url'], keep='first')
        print(f"Số bài trùng lặp đã loại bỏ: {initial_len - len(self.df)}")
        # Clean text
        print("Loại bỏ ký tự đặc biệt và chuyển về chữ thường...")
        self.df['temp_clean'] = self.df[text_column].apply(self.clean_text)
        print("Tách từ Tiếng Việt bằng thư viện underthesea...")
        self.df['temp_segmented'] = self.df['temp_clean'].apply(self.segment_words)
        print("Xóa stopWords...")
        self.df[output_column] = self.df['temp_segmented'].apply(self.remove_stopwords)

        # drop các cột tạm thời đã sử dụng để xử lý
        self.df.drop(columns=['temp_clean', 'temp_segmented'], inplace=True)
        
        # Xóa các dòng có kết quả sau khi xử lý là rỗng
        self.df = self.df[self.df[output_column].str.strip() != ""]

        print(f"Kích thước dữ liệu sau khi tiền xử lý: {self.df.shape}")
        return self.df
    
    def filter_by_length(self, title_range=(2, 12), abstract_range=(5, 30)):
        """
        Loại bỏ các bài báo có độ dài tiêu đề hoặc tóm tắt không phù hợp.
        
        :param title_range: Tuple (min, max) số lượng từ cho tiêu đề.
        :param abstract_range: Tuple (min, max) số lượng từ cho tóm tắt.
        """
        if self.df is None:
            print("Chưa có dữ liệu để lọc.")
            return

        initial_len = len(self.df)
        print(f"Lọc nhiễu dựa trên độ dài văn bản...")

        def count_words(text):
            if not isinstance(text, str): return 0
            return len(text.split())

        self.df['title_count'] = self.df['title_clean'].apply(count_words)
        self.df['abstract_count'] = self.df['abstract_clean'].apply(count_words)

        mask = (
            (self.df['title_count'] >= title_range[0]) & 
            (self.df['title_count'] <= title_range[1]) &
            (self.df['abstract_count'] >= abstract_range[0]) & 
            (self.df['abstract_count'] <= abstract_range[1])
        )
        
        self.df = self.df[mask].copy()
        
        self.df.drop(columns=['title_count', 'abstract_count'], inplace=True)

        removed = initial_len - len(self.df)
        print(f"Loại bỏ {removed} bài báo do độ dài không phù hợp.")
        print(f"Dữ liệu còn lại: {len(self.df)} bài.")
        
        return self.df
    
    def save_data(self, dataframe=None, output_path="../data/processed/news_data_clean.csv"):
        """
        Lưu dữ liệu đã tiền xử lý ra file CSV mới.
        """
        df_to_save = dataframe if dataframe is not None else self.df
        df_to_save.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Đã lưu dữ liệu sạch tại: {output_path}")