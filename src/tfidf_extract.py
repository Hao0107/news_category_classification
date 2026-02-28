import math
import numpy as np
from collections import Counter, defaultdict

class CustomTFIDF:
    def __init__(self):
        """
        Khởi tạo bộ trích xuất đặc trưng TF-IDF.
        """
        self.vocab = {}          # Lưu trữ từ vựng: {từ: chỉ_số_cột}
        self.idf_weights = {}    # Lưu trọng số IDF của từng từ
        self.feature_names = []  # Danh sách từ vựng theo thứ tự cột

    def fit(self, documents):
        """
        Học từ vựng (Vocabulary) và tính toán trọng số IDF từ tập dữ liệu huấn luyện.
        :param documents: List/Series chứa các đoạn văn bản đã tiền xử lý.
        """
        print("[*] Đang xây dựng từ vựng và tính toán IDF...")
        total_docs = len(documents)
        df_counts = defaultdict(int)
        
        unique_words_set = set()
        
        # Đếm Document Frequency (DF): Số lượng văn bản chứa mỗi từ
        for doc in documents:
            if not isinstance(doc, str):
                continue
            
            # Tách từ bằng khoảng trắng (dữ liệu đã được gạch dưới bởi underthesea)
            words = doc.split()
            unique_words = set(words)
            
            for word in unique_words:
                df_counts[word] += 1
                unique_words_set.add(word)
                
        # Sắp xếp từ vựng theo alphabet để cố định thứ tự cột
        self.feature_names = sorted(list(unique_words_set))
        self.vocab = {word: idx for idx, word in enumerate(self.feature_names)}
        
        # Tính toán IDF cho từng từ trong từ vựng
        for word, df in df_counts.items():
            # Công thức Smooth IDF
            self.idf_weights[word] = math.log((1 + total_docs) / (1 + df)) + 1
            
        print(f"   [+] Hoàn tất! Kích thước từ vựng (Vocabulary): {len(self.vocab)} từ.")
        return self

    def transform(self, documents):
        """
        Chuyển đổi văn bản thành ma trận số TF-IDF.
        """
        print("[*] Đang chuyển đổi văn bản thành ma trận TF-IDF...")
        num_docs = len(documents)
        num_features = len(self.vocab)
        
        # Khởi tạo ma trận rỗng (dense matrix)
        tfidf_matrix = np.zeros((num_docs, num_features), dtype=np.float32)
        
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                continue
                
            words = doc.split()
            total_words = len(words)
            
            if total_words == 0:
                continue
                
            # Đếm tần suất xuất hiện của từ trong 1 văn bản
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in self.vocab:
                    # Tính TF
                    tf = count / total_words
                    # Lấy IDF đã học được
                    idf = self.idf_weights[word]
                    # Ghi vào ma trận
                    col_idx = self.vocab[word]
                    tfidf_matrix[i, col_idx] = tf * idf
                    
        # BƯỚC QUAN TRỌNG: Chuẩn hóa L2 (L2 Normalization)
        # Ép độ dài của mỗi vector bài báo về 1 để các bài báo dài/ngắn được đánh giá công bằng
        print("   [+] Đang chuẩn hóa L2 cho ma trận...")
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        # Tránh lỗi chia cho 0 với các dòng trống
        norms[norms == 0] = 1 
        tfidf_matrix = tfidf_matrix / norms
            
        print(f"   [+] Hoàn tất tạo ma trận: Kích thước {tfidf_matrix.shape}")
        return tfidf_matrix

    def fit_transform(self, documents):
        """Kết hợp học và chuyển đổi trong 1 bước."""
        self.fit(documents)
        return self.transform(documents)
        
    def get_feature_names(self):
        """Trả về danh sách từ vựng."""
        return self.feature_names