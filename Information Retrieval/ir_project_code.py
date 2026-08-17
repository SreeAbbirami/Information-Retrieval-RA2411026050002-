import os
import re
import sys
import math
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ====================================================================
# BENCHMARK CORPUS: 10 Research Papers / Documents
# ====================================================================
documents = {
    "Doc 1": "Deep learning architectures for natural language processing and text classification.",
    "Doc 2": "Information retrieval systems utilizing vector space models and tf-idf weighting.",
    "Doc 3": "Neural network approaches to automated document classification and similarity ranking.",
    "Doc 4": "Evaluating search engines using cosine similarity scores and term frequency analysis.",
    "Doc 5": "Natural language processing techniques for stop word removal and stemming algorithms.",
    "Doc 6": "Applying convolutional networks to image classification and computer vision tasks.",
    "Doc 7": "Efficient indexing algorithms for large scale information retrieval databases.",
    "Doc 8": "Query optimization techniques in classical vector space retrieval models.",
    "Doc 9": "Comparative analysis of Porter stemming vs lemmatization in text mining execution time.",
    "Doc 10": "Transformer models and self attention mechanisms in modern natural language processing.",
}

# ====================================================================
# MODULE 1: CORPUS SETUP & PREPROCESSING PIPELINE
# ====================================================================
def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    cleaned_tokens = [
        stemmer.stem(word) for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    return " ".join(cleaned_tokens)

processed_corpus = {doc_id: preprocess_text(text) for doc_id, text in documents.items()}

print("=== OUTPUT: PREPROCESSING RESULTS ===")
for doc_id, text in list(documents.items())[:3]:
    print(f"[{doc_id}] Raw          : {text}")
    print(f"[{doc_id}] Preprocessed : {processed_corpus[doc_id]}")
    print("-" * 75)

# ====================================================================
# MODULE 2: TF-IDF FEATURE MATRIX GENERATION
# ====================================================================
vectorizer = TfidfVectorizer()
corpus_list = list(processed_corpus.values())
doc_ids = list(processed_corpus.keys())
tfidf_matrix = vectorizer.fit_transform(corpus_list)
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=doc_ids,
    columns=feature_names
)

print("\n=== OUTPUT: TF-IDF MATRIX SUMMARY ===")
print(f"Total Unique Vocabulary Features : {len(feature_names)}")
print(f"Matrix Dimension (Docs x Terms)  : {tfidf_matrix.shape}\n")
sample_terms = [t for t in ['vector', 'space', 'model', 'tfidf', 'classifi',
                             'natur', 'languag', 'retriev', 'neural', 'network']
                if t in feature_names]
print("Sample TF-IDF Matrix Weights (rounded):")
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', None)
print(tfidf_df[sample_terms].head(5).round(4).to_string())

# ====================================================================
# MODULE 3: QUERY PROCESSING & COSINE SIMILARITY RANKING
# ====================================================================
def search_and_rank(user_query: str, top_k: int = 5):
    processed_q = preprocess_text(user_query)
    query_vector = vectorizer.transform([processed_q])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(scores)[::-1]
    print(f"\nRAW QUERY       : '{user_query}'")
    print(f"PROCESSED QUERY : '{processed_q}'")
    print(f"{'Rank':<6}{'Doc ID':<10}{'Score':<10}{'Document Snippet'}")
    print("-" * 75)
    doc_values = list(documents.values())
    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        doc_id = doc_ids[idx]
        score = scores[idx]
        snippet = doc_values[idx][:50] + ("..." if len(doc_values[idx]) > 50 else "")
        print(f"{rank:<6}{doc_id:<10}{score:<10.4f}{snippet}")

search_and_rank("vector space models and tf-idf weighting", top_k=3)
search_and_rank("natural language processing", top_k=4)

# ====================================================================
# MODULE 4: PERFORMANCE EVALUATION METRICS
# ====================================================================
def evaluate_pipeline():
    raw_words = re.sub(r'[^a-zA-Z\s]', '', " ".join(documents.values()).lower()).split()
    unique_raw = set(raw_words)
    processed_vocab = set(feature_names)
    reduction = ((len(unique_raw) - len(processed_vocab)) / len(unique_raw)) * 100
    total_processed_words = sum(len(preprocess_text(t).split()) for t in documents.values())
    print("\n=== PIPELINE EVALUATION METRICS ===")
    print(f"Total Raw Corpus Word Count        : {len(raw_words)}")
    print(f"Total Processed Corpus Word Count  : {total_processed_words}")
    print(f"Unique Raw Vocabulary Terms        : {len(unique_raw)}")
    print(f"Processed Features (TF-IDF)        : {len(processed_vocab)}")
    print(f"Vocabulary Dimension Reduction     : {reduction:.2f}%")
    print("=" * 65)

evaluate_pipeline()

# ====================================================================
# MODULE 5: INTEGRATED OOP CLASS IMPLEMENTATION (IRSystem)
# ====================================================================
class IRSystem:
    def __init__(self, doc_dict):
        nltk.download('stopwords', quiet=True)
        self.raw_documents = doc_dict
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.processed_corpus = {
            doc_id: self._preprocess(text)
            for doc_id, text in self.raw_documents.items()
        }
        self.doc_ids = list(self.processed_corpus.keys())
        self.tfidf_matrix = self.vectorizer.fit_transform(
            list(self.processed_corpus.values())
        )
        self.feature_names = self.vectorizer.get_feature_names_out()

    def _preprocess(self, text: str) -> str:
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = text.split()
        cleaned = [
            self.stemmer.stem(w)
            for w in tokens if w not in self.stop_words and len(w) > 1
        ]
        return " ".join(cleaned)

    def search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        processed_q = self._preprocess(query)
        q_vec = self.vectorizer.transform([processed_q])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            results.append({
                "Rank": rank,
                "Doc ID": self.doc_ids[idx],
                "Score": round(scores[idx], 4),
                "Text": self.raw_documents[self.doc_ids[idx]]
            })
        return pd.DataFrame(results)

engine = IRSystem(documents)
print("\n=== CLASS SEARCH RESULTS ===")
query = "neural network text classification"
print(f"Query: '{query}'  (top_k=2)")
res = engine.search(query, top_k=2)
pd.set_option('display.max_colwidth', 70)
print(res.to_string(index=False))
