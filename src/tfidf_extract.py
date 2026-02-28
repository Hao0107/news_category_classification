import math
import numpy as np
from collections import Counter, defaultdict

class CustomTFIDF:
    def __init__(self, min_df=5):
        """
        Khởi tạo thuật toán TF-IDF.
        :param min_df: Loại bỏ các từ xuất hiện trong ÍT HƠN min_df bài báo.
                       (Tham số này giúp giảm từ 17.727 từ xuống còn ~7.000 từ).
        """
        self.min_df = min_df
        self.vocab = {}           # Lưu mapping: {từ_vựng: chỉ_số_cột}
        self.idf_weights = {}     # Trọng số Inverse Document Frequency
        self.feature_names = []   # Danh sách từ vựng chuẩn

    def fit(self, documents):
        """
        Quét toàn bộ văn bản để xây dựng từ điển và tính IDF.
        """
        print(f"Đang học từ vựng từ {len(documents)} bài báo...")
        df_counts = defaultdict(int)
        total_docs = len(documents)

        # Tính Document Frequency (DF) cho từng từ
        for doc in documents:
            if not isinstance(doc, str):
                continue

            # Dùng set() để đảm bảo mỗi từ chỉ đếm 1 lần cho 1 bài báo
            unique_words = set(doc.split())
            for word in unique_words:
                df_counts[word] += 1

        # Áp dụng màng lọc min_df để loại bỏ "Từ cực hiếm"
        filtered_words = {word: df for word, df in df_counts.items() if df >= self.min_df}

        # Cố định thứ tự cột bằng cách sắp xếp alphabet
        self.feature_names = sorted(list(filtered_words.keys()))
        self.vocab = {word: idx for idx, word in enumerate(self.feature_names)}

        # Tính toán trọng số IDF (Sử dụng Smooth IDF)
        # Công thức: IDF(t) = log((1 + Tổng_số_bài) / (1 + Số_bài_chứa_từ_t)) + 1
        for word, df in filtered_words.items():
            self.idf_weights[word] = math.log((1 + total_docs) / (1 + df)) + 1

        print(f"Đã giữ lại {len(self.vocab)} từ vựng cốt lõi (Bỏ qua từ xuất hiện < {self.min_df} lần).")
        return self

    def transform(self, documents):
        """
        Chuyển đổi văn bản thành ma trận toán học TF-IDF.
        """
        print("Đang chuyển đổi văn bản thành ma trận TF-IDF...")
        num_docs = len(documents)
        num_features = len(self.vocab)

        # Khởi tạo ma trận rỗng bằng Numpy
        tfidf_matrix = np.zeros((num_docs, num_features), dtype=np.float32)

        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                continue

            words = doc.split()
            total_words = len(words)
            if total_words == 0:
                continue

            # Đếm số lần xuất hiện của từng từ trong 1 bài báo cụ thể
            word_counts = Counter(words)

            for word, count in word_counts.items():
                if word in self.vocab:
                    # Tính Term Frequency (TF)
                    # count: số lần xuất hiện của từ, total_words: tổng số từ trong bài
                    tf = count / total_words

                    # Lấy IDF đã tính ở hàm fit
                    idf = self.idf_weights[word]

                    # Ghi kết quả vào đúng tọa độ [dòng_i, cột_word]
                    col_idx = self.vocab[word]
                    tfidf_matrix[i, col_idx] = tf * idf

        # Chuẩn hóa L2 (L2 Normalization)
        # Ép độ dài của mỗi vector bài báo về 1.
        print("Đang chuẩn hóa L2 (Euclidean norm) cho ma trận...")
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Tránh lỗi chia cho 0
        tfidf_matrix = tfidf_matrix / norms

        print(f"Hoàn tất! Kích thước ma trận cuối cùng: {tfidf_matrix.shape}")
        return tfidf_matrix

    def fit_transform(self, documents):
        """
        Chạy liên tiếp 2 quá trình.
        """
        self.fit(documents)
        return self.transform(documents)