from collections import Counter
import concurrent.futures
import os
import json
import pickle
import time
import threading
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from datasketch import MinHash, MinHashLSH

nltk.download("stopwords")

MAX_OFFLOAD_WORKERS = 5
MAX_PARTIAL_INDEX_POSTINGS = 1000000
IMPORTANT_TOKEN_WEIGHT_MULTIPLIER = 3
SIMILARITY_THRESHOLD = 0.7
IMPORTANTS_TAGS = ('h1', 'h2', 'h3', 'b', 'strong', 'title')
TEXT_TAGS = ('p', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'cite', 'q')


class Indexer:
    def __init__(self, dev_folder_path, index_folder_path, url_map_path) -> None:
        self.index_folder = index_folder_path
        self.url_map_path = url_map_path
        self.document_generator = self.get_json_files(dev_folder_path)
        
        self.inv_index = {}
        self.doc_urls = {}
        
        self.lock = threading.Lock()
        self.offload_thread = None
        self.offload_index = threading.Event()

        self.docid_count = 1
        self.postings_count = 0
        
        self.minhashes = set()
        self.lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=128)  # Initialize LSH

        self.unique_words = set()
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        
        self.running = True
    
    @staticmethod
    def clean_index():
        file_paths = [f"index/index_range_{range_key}.pkl" for range_key in ("a-f", "g-l", "m-r", "s-z", "0-9")] + ["index/urls.pkl"]
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"Found existing index file: {file_path} - deleting")
                os.remove(file_path)
     
    def build_index(self):
        self.clean_index()
        
        start_time = time.time()
        
        self.offload_thread = threading.Thread(target=self._offload_index_worker)
        self.offload_thread.start()
    
        self._process_documents()
        
        self.offload_thread.join()
        
        self._build_url_map()
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total execution time: {elapsed_time/60:.5f} minutes")
        
    def _process_documents(self):
        for document in self.document_generator:
            self._parse_document(document)
        print(f"\n[PROCESSING THREAD] Offloading leftover index with {self.postings_count} postings left\n")
        self.offload_index.set()
        self.running = False

    def _parse_document(self, document):
        start_time = time.time()
        docid = self.docid_count
        tokens = Counter()
        try:
            with open(document, 'r') as file:
                json_data = json.load(file)
            url = json_data["url"]
            content = json_data.get("content", "")
            if content:
                soup_start_time = time.time()
                soup = BeautifulSoup(content, "xml")
                text_content = soup.get_text()
                soup_time = time.time() - soup_start_time
                if soup_time > 1:
                    print(f"\tsoup time took too long: {soup_time:.6f}s")
                
                dup_start_time = time.time()
                if self._is_near_duplicate(text_content, docid):
                    print(f"document {docid} (dup)")
                    return
                dup_time = time.time() - dup_start_time
                if dup_time > 1:
                    print(f"\tduplicate time took too long: {dup_time:.6f}s")
                
                self.doc_urls[docid] = url
                print(f"document {docid}", end="")
                
                token_start_time = time.time()
                for tag in soup.find_all(TEXT_TAGS + IMPORTANTS_TAGS):
                    tag_tokens = self._tokenize_tag(tag, is_important=(tag.name in IMPORTANTS_TAGS))
                    tokens.update(tag_tokens)
                token_time = time.time() - token_start_time
                if token_time > 1:
                    print(f"\ttoken time took too long: {token_time:.6f}s")
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f" - {elapsed_time:.6f}s")
                    
                self._update_index(tokens, docid)
            else:
                print(f"document {docid} had no content")
        except FileNotFoundError:
            print(f"File {document} not found.")
        except Exception as e:
            print(f"Unexpected error processing document {document}: {e}")
        finally:
            self.docid_count += 1
            
            if self.postings_count >= MAX_PARTIAL_INDEX_POSTINGS:
                if not self.offload_index.is_set():
                    print(f"\n[PROCESSING THREAD] Postings limit exceeded: {self.postings_count}\n")
                    self.postings_count = 0
                    self.offload_index.set()
                else:
                    print("event is already set")
        
    def _is_near_duplicate(self, text_content, docid):
        shingles = set(ngrams(text_content.split(), 3))
        m = MinHash(num_perm=128)
        for shingle in shingles:
            m.update(" ".join(shingle).encode('utf8'))
        if len(self.lsh.query(m)) > 0:
            return True
        self.lsh.insert(docid, m)
        return False
    
    def _tokenize_tag(self, tag, is_important=False):
        tokens = []
        text = tag.get_text()
        for word in word_tokenize(text):
            if word.isalnum():
                token = self.stemmer.stem(word.lower())
                if token not in self.stopwords and is_important:
                    tokens.extend([token] * IMPORTANT_TOKEN_WEIGHT_MULTIPLIER)  # Increase weight of important tokens
                else:
                    tokens.append(token)
        return tokens
    
    def _update_index(self, tokens, docid):
        with self.lock:
            for token, count in tokens.items():
                posting = (docid, count)
                range_key = self.get_range_key(token[0])
                if range_key not in self.inv_index:
                    self.inv_index[range_key] = {}
                if token not in self.inv_index[range_key]:
                    self.inv_index[range_key][token] = [posting]
                else:
                    self.inv_index[range_key][token].append(posting)
                self.postings_count += 1
                self.unique_words.add(token)  
    
    def _offload_index_worker(self):
        while self.running:
            print("\n[OFFLOAD THREAD] Waiting for offload event\n")
            self.offload_index.wait() # wait for event
            self._offload_partial_index()
            self.offload_index.clear()
                    
    def _offload_partial_index(self):
        print("\n[OFFLOAD THREAD] Begin offload\n")
        start_time = time.time()
        
        with self.lock:
            offload_copy = self.inv_index.copy()
            self.inv_index.clear()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_OFFLOAD_WORKERS) as executor:
            futures = []
            for range_key, index in offload_copy.items():
                futures.append(executor.submit(self._process_partial_index, range_key, index))
            concurrent.futures.wait(futures)
            
        elapsed_time = time.time() - start_time
        print(f"\n[OFFLOAD THREAD] Offload completed in {elapsed_time}s\n")

    def _process_partial_index(self, range_key, index):
        range_file = os.path.join(self.index_folder, f"index_range_{range_key}.pkl")
        if os.path.exists(range_file):
            with open(range_file, 'rb') as f:
                existing_index = pickle.load(f)
                for term, postings in index.items():
                    if term in existing_index:
                        existing_index[term].extend(postings)
                    else:
                        existing_index[term] = postings
            with open(range_file, 'wb') as f:
                pickle.dump(existing_index, f)
        else:
            with open(range_file, 'wb') as f:
                pickle.dump(index, f)
             
    def _build_url_map(self):
        with open(self.url_map_path, 'wb') as f:
            pickle.dump(self.doc_urls, f)

    def print_report(self):
        print(f"Number of unique words: {len(self.unique_words)}")
    
    @staticmethod
    def get_range_key(first_char):
        if 'a' <= first_char <= 'f':
            return 'a-f'
        elif 'g' <= first_char <= 'l':
            return 'g-l'
        elif 'm' <= first_char <= 'r':
            return 'm-r'
        elif 's' <= first_char <= 'z':
            return 's-z'
        else:
            return '0-9'

    @staticmethod
    def get_json_files(directory):
        json_files = []
        for root, dirs, files in os.walk(directory, topdown=False):
            for file in files:
                json_files.append(os.path.join(root, file))
        return json_files


if __name__ == "__main__":
    dev_folder_path = r"C:\Users\Owner\school\CS121\Assignments\rsrc\DEV"
    indexer = Indexer(dev_folder_path, index_folder_path=r".\index", url_map_path=r".\index\urls.pkl")
    indexer.build_index()
