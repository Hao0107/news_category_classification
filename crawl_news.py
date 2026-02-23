import requests
from bs4 import BeautifulSoup
import time
import csv
import random
import yaml
import os
import threading
import concurrent.futures
from urllib.parse import urlparse

# --- CONFIGURATION ---
class ConfigManager:
    def __init__(self, config_path="./sites_config.yaml"):
        '''
            Khoi tao ConfigManager, doc file cau hinh va luu vao self.sites va self.categories
        
        :param config_path: Duong dan toi file cau hinh yaml chua thong tin ve cac trang tin tuc va the loai can crawl
        '''
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.sites = config.get('sites', {})
            self.categories = config.get('categories', {})
        except Exception as e:
            print(f" Could not load {config_path}: {e}")
            exit(1)

# --- STORAGE MANAGER ---
class CSVStorageManager:
    def __init__(self, output_file="./data/large_news_dataset.csv"):
        self.output_file = output_file
        
        # su dung lock de tranh loi khi ghi file tu nhieu thread cung luc
        self.lock = threading.Lock()
        self.total_saved = 0
        self._setup_file()

    def _setup_file(self):
        '''
            Tao file csv va viet header neu chua ton tai. 
            Neu file da ton tai, giu nguyen noi dung de tiep tuc ghi them du lieu moi.
        '''
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['category', 'source', 'title', 'abstract', 'url'])

    def save(self, data_dict):
        '''
            Luu mot bai viet vao file csv. 
            Su dung lock de dam bao chi 1 thread duoc phep ghi vao file tai 1 thoi diem.
            Moi bai viet duoc luu se tang bien dem self.total_saved.
        :param data_dict: Mot dict chua thong tin ve bai viet, voi cac key: category, source, title, abstract, url
        '''
        with self.lock:
            with open(self.output_file, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    data_dict['category'], data_dict['source'], 
                    data_dict['title'], data_dict['abstract'], data_dict['url']
                ])
            self.total_saved += 1

# --- ARTICLE PARSER ---
class ArticleParser:
    def __init__(self):
        '''
            Khoi tao ArticleParser, dat header de su dung khi gui request den cac trang tin tuc.
        '''
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7'
        }

    def get_domain(self, url):
        return urlparse(url).netloc.replace('www.', '')

    def get_soup(self, url):
        '''
            Gui request den url va tra ve doi tuong BeautifulSoup de phan tich HTML.
            Tra ve None neu co loi xay ra trong qua trinh ket noi hoac phan tich.
        
        :param url: URL cua trang tin tuc can phan tich
        '''
        time.sleep(random.uniform(0.5, 1.5))
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            pass
        return None

    def extract_data(self, url, category, site_rules):
        '''
            Trich xuat thong tin tu mot bai viet dua tren url, the loai va quy tac cua trang.
            Neu trich xuat thanh cong, tra ve mot dict chua category, source, title, abstract va url.
            Neu co loi xay ra hoac du lieu khong hop le, tra ve None.
        :param url: URL cua bai viet can trich xuat
        :param category: The loai cua bai viet (vd: the thao, giai tri,...)
        :param site_rules: Mot dict chua cac selector de tim tieu de va tom tat tren trang tin tuc tuong ung
        '''
        soup = self.get_soup(url)
        if not soup: return None

        title_tag = soup.select_one(site_rules['title_selector'])
        abs_tag = soup.select_one(site_rules['abstract_selector'])

        if title_tag and abs_tag:
            return {
                'category': category,
                'source': self.get_domain(url),
                'title': title_tag.get_text(strip=True).replace('\n', ' '),
                'abstract': abs_tag.get_text(strip=True).replace('\n', ' '),
                'url': url
            }
        return None

# --- CRAWL ORCHESTRATOR ---
class CrawlOrchestrator:
    def __init__(self, config_manager, storage_manager, parser):
        '''
            Khoi tao CrawlOrchestrator voi cac thanh phan: 
                config_manager de doc cau hinh, 
                storage_manager de luu du lieu 
                va parser de trich xuat thong tin tu bai viet.
            CrawlOrchestrator se dung cac thanh phan nay de thuc hien viec 
            crawl du lieu tu cac trang tin tuc khac nhau mot cach dong bo va hieu qua.
        
        :param config_manager: Mot instance cua ConfigManager, da duoc khoi tao va doc cau hinh tu file yaml
        :param storage_manager: Mot instance cua CSVStorageManager, da duoc khoi tao va san sang de luu du lieu vao file csv
        :param parser: Mot instance cua ArticleParser, da duoc khoi tao va san sang de trich xuat thong tin tu bai viet
        '''
        self.config = config_manager
        self.storage = storage_manager
        self.parser = parser

    def _worker_task(self, url, category, rules):
        '''
            Task duoc thuc thi boi moi thread trong qua trinh crawl.
            Nhiem vu cua task la trich xuat thong tin tu bai viet va luu vao file csv neu du lieu hop le.
            Neu co loi xay ra trong qua trinh trich xuat hoac luu du lieu, 
            task se in ra loi va tiep tuc thuc hien cac task con lai ma khong bi gian doan.
        
        :param url: URL cua bai viet can trich xuat
        :param category: The loai cua bai viet (vd: the thao, giai tri,...)
        :param rules: Mot dict chua cac selector de tim tieu de va tom tat tren trang tin tuc tuong ung
        '''
        data = self.parser.extract_data(url, category, rules)
        if data:
            self.storage.save(data)

    def run(self, max_pages=10, max_workers=8):
        print("=== COLLECTING URLS ===")
        tasks = []
        
        for category, base_urls in self.config.categories.items():
            print(f"\n Scanning Category: {category}")
            for base_url in base_urls:
                clean_base_url = base_url.rstrip('/')
                domain = self.parser.get_domain(clean_base_url)
                rules = self.config.sites.get(domain)
                
                if not rules: 
                    print(f" [Skip] No configuration found for {domain}")
                    continue
                print(f"   -> Target: {domain} | Category Link: {clean_base_url}")
                
                # Pagination loop: tu page 1 den max_pages, tao url va lay danh sach link bai viet tren tung trang
                for page in range(1, max_pages + 1):
                    # Tao URL cho trang hien tai dua tren template va page number
                    if page == 1:
                        current_url = clean_base_url
                    else:
                        template = rules.get('pagination_template', "{base_url}")
                        current_url = template.format(base_url=clean_base_url, page=page)
                    # IN RA MAN HINH: Dang tai trang nao
                    print(f"      - Fetching Page {page}: {current_url}")

                    # Lay soup cua trang hien tai de tim link bai viet
                    soup = self.parser.get_soup(current_url)
                    if not soup: 
                        print(f"Failed to retrieve HTML (Timeout or Blocked).")
                        continue

                    # Tim tat ca the link bai viet tren trang hien tai dua tren selector trong cau hinh
                    links = soup.select(rules.get('article_link_selector', 'a'))
                    print(f" > Found {len(links)} links on this page.")
                    
                    # Duyet tung link bai viet, chuan hoa URL va them vao danh sach task de trich xuat sau nay
                    for link_tag in links:
                        if 'href' in link_tag.attrs:
                            article_url = link_tag['href']
                            
                            # Xu ly URL bai viet: neu la URL tuong doi, chuan hoa thanh URL tuyet doi
                            if not article_url.startswith('http'):
                                article_url = f"https://{domain}{article_url}"
                            
                            # Them task moi vao danh sach, task gom URL bai viet, 
                            # the loai va quy tac de trich xuat tu trang tuong ung
                            tasks.append((article_url, category, rules))

        # Su dung set de loc cac task trung lap (cung URL bai viet) de tranh viec trich xuat va luu du lieu trung lap
        unique_tasks = []
        seen_urls = set()
        
        for task in tasks:
            url = task[0]  # The article_url is the first item in the tuple
            if url not in seen_urls:
                seen_urls.add(url)
                unique_tasks.append(task)
                
        tasks = unique_tasks

        print(f"\nFound {len(tasks)} unique articles to download.")
        print(f"=== CONCURRENT DOWNLOADING ({max_workers} threads) ===")
        
        # su dung ThreadPoolExecutor de thuc hien cac task trich xuat du lieu tu cac bai viet mot cach dong bo va hieu qua
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._worker_task, url, cat, rules) for url, cat, rules in tasks]
            
            # doi cho tat ca cac task hoan thanh, neu co loi xay ra trong qua trinh thuc hien task, 
            # in ra loi va tiep tuc thuc hien cac task con lai
            concurrent.futures.wait(futures)
            
        print(f"\n=== CRAWL COMPLETE. TOTAL SAVED: {self.storage.total_saved} ===")
