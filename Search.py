import os
import pickle
from collections import defaultdict
from math import log10, sqrt
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from Indexer import Indexer

class SearchEngine:
    def __init__(self, index_folder_path, urls_pkl_path):
        self.index_folder_path = index_folder_path
        self.stemmer = PorterStemmer()
        self.doc_urls = self._load_urls(urls_pkl_path)
        self.status = "success"
        if not self.doc_urls:
            self.status = "fail"
        self.num_docs = len(self.doc_urls)
        #self.doc_lengths = self._compute_doc_lengths()

    def _load_urls(self, urls_pkl_path):
        file_paths = [f"index/index_range_{range_key}.pkl" for range_key in ("a-f", "g-l", "m-r", "s-z", "0-9")] + ["index/urls.pkl"]
        for file_path in file_paths:
            if not os.path.exists(file_path):
                return []
            try:
                with open(urls_pkl_path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                print(f"File {urls_pkl_path} not found.")

    def _get_tfidf_score(self, term_freq, num_doc_with_token):
        # Term Frequency (TF)
        tf = 1 + log10(term_freq) if term_freq > 0 else 0
        # Inverse Document Frequency (IDF)
        idf = self.num_docs / num_doc_with_token if num_doc_with_token != 0 else 0
        return tf * idf

    def _get_postings(self, token):
        index_range_file =  os.path.join(self.index_folder_path, f"index_range_{Indexer.get_range_key(token[0])}.pkl")
        with open(index_range_file, 'rb') as f:
            index = pickle.load(f)
            return index.get(token, [])

    
    def search(self, query):
        stemmed_query = [self.stemmer.stem(word.lower()) for word in word_tokenize(query)]
        
        token_postings = {token: self._get_postings(token) for token in stemmed_query} # {token1: [posting1, posting2]}
        
        docid_sets = [set([docid for docid, _ in postings]) for postings in token_postings.values()]
        print(docid_sets)
        if len(docid_sets) == 0:
            return []
        
        common_docids = set.intersection(*docid_sets) # AND logic
        
        query_vector = {token: self._get_tfidf_score(stemmed_query.count(token), len(postings)) for token, postings in token_postings.items()}
        
        scores = defaultdict(float)
        query_norm = sqrt(sum(val ** 2 for val in query_vector.values()))
        
        for docid in common_docids:
            doc_vector = {token: self._get_tfidf_score(tf, len(postings)) for token, postings in token_postings.items() for curr_docid, tf in postings if curr_docid == docid}
            
            dot_product = sum(query_vector[token] * doc_vector.get(token, 0) for token in query_vector)
            doc_norm = sqrt(sum(val ** 2 for val in doc_vector.values()))
            cosine_similarity = dot_product / (query_norm * doc_norm) if (query_norm * doc_norm) != 0 else 0
            scores[docid] = cosine_similarity
        
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.doc_urls[doc_id], score) for doc_id, score in ranked_docs]
    
    @staticmethod
    def display_results(results):
        if not results:
            print("No results found.")
            return
        for i, (url, score) in enumerate(results, start=1):
            print(f"{i}. {url} - Score: {score:.3f}")
