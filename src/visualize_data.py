from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud

# Bỏ qua các cảnh báo không quan trọng của Seaborn
warnings.filterwarnings("ignore")

class NewsDataVisualizer:
    def __init__(self, csv_path):
        """
        Khoi tao lop NewsDataVisualize.
        :param csv_path: Đường dẫn tới file CSV chứa dữ liệu.
        """
        print(f"load data tu: {csv_path}")
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Tải thành công! Kích thước dữ liệu: {self.df.shape[0]} dòng, {self.df.shape[1]} cột.")
            
            # Xóa các dòng bị trùng lặp URL (nếu có sót lại)
            initial_len = len(self.df)
            if 'url' in self.df.columns:
                self.df.drop_duplicates(subset=['url'], inplace=True)
            if len(self.df) < initial_len:
                print(f"Da xoa {initial_len - len(self.df)} dong trung lap URL.")
                
        except Exception as e:
            print(f"Khong doc duoc file: {e}")
            self.df = None

    def plot_category_distribution(self):
        """
        Vẽ biểu đồ phân bố số lượng bài viết theo từng thể loại (Category).
        """
        if self.df is None or 'category' not in self.df.columns:
            print("Không tìm thấy cột 'category' trong dữ liệu.")
            return

        plt.figure(figsize=(12, 6))
        order = self.df['category'].value_counts().index
        ax = sns.countplot(data=self.df, y='category', order=order, palette='viridis')
        
        plt.title('Phân Bố Số Lượng Bài Viết Theo Thể Loại', fontsize=16, fontweight='bold')
        plt.xlabel('Số Lượng Bài Viết', fontsize=12)
        plt.ylabel('Thể Loại', fontsize=12)
        
        for container in ax.containers:
            ax.bar_label(container, padding=5)
            
        plt.tight_layout()
        plt.show()

    def plot_source_distribution(self):
        """
        Vẽ biểu đồ xem mỗi nguồn báo (Source) đóng góp bao nhiêu bài viết.
        """
        if self.df is None or 'source' not in self.df.columns:
            print("Không tìm thấy cột 'source' trong dữ liệu.")
            return

        plt.figure(figsize=(10, 5))
        order = self.df['source'].value_counts().index
        ax = sns.countplot(data=self.df, x='source', order=order, palette='Set2')
        
        plt.title('Tỉ Trọng Dữ Liệu Theo Nguồn Báo', fontsize=14, fontweight='bold')
        plt.xlabel('Nguồn Báo', fontsize=12)
        plt.ylabel('Số Lượng Bài', fontsize=12)
        plt.xticks(rotation=45)
        
        for container in ax.containers:
            ax.bar_label(container, padding=3)
            
        plt.tight_layout()
        plt.show()

    def plot_text_length(self, text_column='abstract'):
        """
        Vẽ biểu đồ phân bố độ dài (số lượng từ) của một cột văn bản.
        Mặc định phân tích cột 'abstract' (tóm tắt). Bạn có thể truyền vào 'title' hoặc 'text_clean' sau này.
        """
        if self.df is None or text_column not in self.df.columns:
            print(f"[Lỗi] Không tìm thấy cột '{text_column}' để phân tích độ dài.")
            return

        # Loại bỏ giá trị NaN (rỗng) trước khi đếm từ
        valid_texts = self.df[text_column].dropna().astype(str)
        word_counts = valid_texts.apply(lambda x: len(x.split()))

        # Vẽ 2 biểu đồ cạnh nhau: Histogram (tổng quan) và Boxplot (tìm điểm dị biệt)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 1. Histogram
        sns.histplot(word_counts, bins=50, kde=True, ax=axes[0], color='royalblue')
        axes[0].set_title(f'Phân Bố Số Lượng Từ Trong "{text_column}"', fontsize=12)
        axes[0].set_xlabel('Số lượng từ')
        axes[0].set_ylabel('Số lượng bài')

        # 2. Boxplot
        sns.boxplot(x=word_counts, ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot Phát Hiện Dị Biệt (Outliers) - "{text_column}"', fontsize=12)
        axes[1].set_xlabel('Số lượng từ')

        plt.tight_layout()
        plt.show()

        # In ra thống kê mô tả
        print(f"\n--- THỐNG KÊ CHI TIẾT ĐỘ DÀI CỘT '{text_column}' ---")
        print(word_counts.describe().round(2))

    def plot_source_category_heatmap(self):
        """
        Vẽ bản đồ nhiệt (Heatmap) thể hiện mối quan hệ giữa Nguồn báo và Thể loại.
        Giúp xem trang báo nào mạnh về chủ đề nào.
        """
        if self.df is None or 'source' not in self.df.columns or 'category' not in self.df.columns:
            print("[Lỗi] Thiếu cột 'source' hoặc 'category'.")
            return

        plt.figure(figsize=(12, 8))
        # Tạo bảng chéo đếm số lượng
        cross_tab = pd.crosstab(self.df['category'], self.df['source'])
        
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
        plt.title('Bản Đồ Nhiệt: Phân Bố Thể Loại Theo Từng Trang Báo', fontsize=14, fontweight='bold')
        plt.xlabel('Nguồn Báo', fontsize=12)
        plt.ylabel('Thể Loại', fontsize=12)
        
        plt.tight_layout()
        plt.show()

    def general_info(self):
        """
        In ra thông tin tổng quan về bộ dữ liệu (Số lượng null, kiểu dữ liệu).
        """
        if self.df is None:
            return
            
        print("\n" + "="*50)
        print(" THÔNG TIN TỔNG QUAN VỀ DỮ LIỆU (DATA INFO)")
        print("="*50)
        self.df.info()
        
        print("\n" + "="*50)
        print(" SỐ LƯỢNG GIÁ TRỊ RỖNG (MISSING VALUES)")
        print("="*50)
        print(self.df.isnull().sum())
        
    def plot_wordcloud_by_category(self, category_name, text_column='title', max_words=100):
        """
        Vẽ Word Cloud cho một thể loại cụ thể.
        Giúp hình dung nhanh các chủ đề/từ khóa đang hot trong thể loại đó.
        
        :param category_name: Tên thể loại muốn vẽ (VD: 'Thể thao', 'Kinh doanh')
        :param text_column: Cột chứa văn bản
        :param max_words: Số lượng từ tối đa hiển thị trên hình (mặc định 100)
        """
        if self.df is None or text_column not in self.df.columns or 'category' not in self.df.columns:
            print("[Lỗi] Thiếu cột dữ liệu cần thiết.")
            return

        df_cat = self.df[self.df['category'] == category_name]
        if df_cat.empty:
            print(f"[Lỗi] Không tìm thấy bài viết nào thuộc thể loại: {category_name}")
            return

        text_data = " ".join(df_cat[text_column].dropna().astype(str).tolist())

        wordcloud = WordCloud(
            width=1200, height=600, 
            background_color='white', 
            max_words=max_words,           
            colormap='viridis',        
            random_state=42
        ).generate(text_data)

        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud: Chủ đề "{category_name}"', fontsize=20, fontweight='bold', pad=20)
        plt.axis('off') 
        plt.tight_layout()
        plt.show()
        
    def plot_vocabulary_stats(self, text_column='title_clean'):
        """
        Phân tích và trực quan hóa tập từ vựng (Vocabulary) của dữ liệu văn bản.
        Giúp định hình chiến lược cắt tỉa từ vựng (max_features, min_df) cho TF-IDF.
        """
        if self.df is None or text_column not in self.df.columns:
            print(f"Không tìm thấy cột '{text_column}' trong dữ liệu.")
            return

        print(f"Đang quét toàn bộ tập dữ liệu trên cột '{text_column}'...")
        
        # Thu thập dữ liệu tần suất
        # DF (Document Frequency): Số lượng bài báo có chứa một từ cụ thể
        doc_frequency = defaultdict(int)
        total_docs = len(self.df)
        
        for text in self.df[text_column].dropna():
            # Dùng set để đếm mỗi từ tối đa 1 lần trên 1 bài báo (cho DF)
            unique_words_in_doc = set(str(text).split())
            for word in unique_words_in_doc:
                doc_frequency[word] += 1
                
        # Sắp xếp từ vựng theo số lượng bài báo giảm dần
        sorted_df = sorted(doc_frequency.items(), key=lambda x: x[1], reverse=True)
        words = [item[0] for item in sorted_df]
        counts = [item[1] for item in sorted_df]
        
        total_vocab = len(words)
        print(f"   [+] Tổng số từ vựng độc lập (Vocabulary Size): {total_vocab:,} từ.")
        print(f"   [+] Từ xuất hiện nhiều nhất: '{words[0]}' (trong {counts[0]} bài báo)")

        # Xem có bao nhiêu từ chỉ xuất hiện đúng 1 lần (Rare words)
        words_appear_once = sum(1 for c in counts if c == 1)
        words_appear_under_5 = sum(1 for c in counts if c < 5)
        
        print(f"Số từ cực hiếm (chỉ xuất hiện 1 lần): {words_appear_once:,} từ (Chiếm {words_appear_once/total_vocab*100:.1f}%)")
        print(f"Số từ hiếm (< 5 bài báo): {words_appear_under_5:,} từ (Chiếm {words_appear_under_5/total_vocab*100:.1f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Biểu đồ 1: Top 20 từ xuất hiện trong nhiều bài báo nhất
        top_n = 20
        sns.barplot(x=counts[:top_n], y=words[:top_n], ax=axes[0], palette='viridis')
        axes[0].set_title(f'Top {top_n} Từ Xuất Hiện Phổ Biến Nhất', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Số lượng bài báo (Document Frequency)')
        axes[0].set_ylabel('Từ vựng')

        # Biểu đồ 2: Đường cong phân bố từ vựng (Zipf's Law)
        # Vì đuôi rất dài, ta dùng thang đo Logarit cho trục X (Thứ hạng từ)
        axes[1].plot(range(1, total_vocab + 1), counts, color='red', linewidth=2)
        axes[1].set_yscale('log') # Thang đo log cho trục Y (Tần suất)
        axes[1].set_xscale('log') # Thang đo log cho trục X (Thứ hạng)
        axes[1].set_title('Phân Bố Tần Suất Từ Vựng (Log-Log Scale)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Thứ hạng của từ (Đã sắp xếp phổ biến -> hiếm)')
        axes[1].set_ylabel('Số lượng bài báo chứa từ đó (Log scale)')
        axes[1].grid(True, which="both", ls="--", alpha=0.5)

        # Thêm đường cắt (Cut-off line) mô phỏng việc bỏ đi các từ xuất hiện < 5 lần
        cut_off_rank = total_vocab - words_appear_under_5
        axes[1].axvline(x=cut_off_rank, color='blue', linestyle='--', label='Ngưỡng loại bỏ từ hiếm (<5 bài)')
        axes[1].legend()

        plt.tight_layout()
        plt.show()