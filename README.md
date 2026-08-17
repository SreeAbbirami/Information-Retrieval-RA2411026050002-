# Information-Retrieval-RA2411026050002-

# 🔎 TF-IDF Based Document Search Engine

A Python-based **Information Retrieval (IR) system** that searches a collection of documents and ranks them according to their relevance to a user's query using **TF-IDF (Term Frequency–Inverse Document Frequency)** and **Cosine Similarity**.

The project implements a complete Information Retrieval pipeline including text preprocessing, TF-IDF feature extraction, query processing, similarity calculation, document ranking, evaluation, and an object-oriented `IRSystem` implementation.

---

## 📌 Project Overview

The system accepts a natural-language search query and retrieves the most relevant documents from a predefined document corpus.

The retrieval process follows these major stages:

```text
User Query
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
Stop-Word Removal
    ↓
Porter Stemming
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity
    ↓
Document Ranking
    ↓
Top-K Relevant Documents
```

The project was developed as part of an **Information Retrieval course assignment** using Python.

---

## ✨ Key Features

* 🔤 Text normalization and preprocessing
* 🧹 Stop-word removal using NLTK
* 🌱 Porter stemming
* 📊 TF-IDF feature extraction
* 🧮 Vector Space Model representation
* 🔍 Natural-language query processing
* 📐 Cosine similarity calculation
* 🏆 Relevance-based document ranking
* 📋 Top-K search results
* 🧱 Object-oriented `IRSystem` implementation
* 📈 Pipeline evaluation metrics
* 📑 Pandas DataFrame-based result presentation
* 💾 Support for exporting ranked results to CSV/Excel through Pandas

---

## 🛠️ Technologies Used

| Technology       | Purpose                                    |
| ---------------- | ------------------------------------------ |
| **Python**       | Core programming language                  |
| **NLTK**         | Stop-word removal and Porter stemming      |
| **Scikit-learn** | TF-IDF vectorization and cosine similarity |
| **Pandas**       | Result processing and tabular presentation |
| **NumPy**        | Numerical operations and ranking           |
| **ReportLab**    | PDF report generation                      |

These libraries and their roles are documented in the project report.

---

## 📂 Project Structure

```text
TF-IDF-Document-Search-Engine/
│
├── ir_project_code.py
├── tfidf_matrix_numerical.py
├── ir.py
├── generate_pdf.py
│
├── documents_dataset/
│   ├── document1.txt
│   ├── document2.txt
│   └── ...
│
├── IR_Project_Documentation.pdf
├── requirements.txt
└── README.md
```

### Main Files

| File / Folder                  | Description                                                |
| ------------------------------ | ---------------------------------------------------------- |
| `ir_project_code.py`           | Main implementation of the Information Retrieval pipeline  |
| `tfidf_matrix_numerical.py`    | Interactive TF-IDF calculation and numerical demonstration |
| `documents_dataset/`           | Additional text documents for corpus expansion             |
| `generate_pdf.py`              | Script used to generate the project documentation          |
| `ir.py`                        | Earlier version of the search engine implementation        |
| `IR_Project_Documentation.pdf` | Complete project documentation                             |

The submitted files are listed in Section 9 of the project report.

---

# ⚙️ How the System Works

## 1. Corpus Creation

The project initially uses **10 manually prepared documents** covering topics such as:

* Natural Language Processing
* Information Retrieval
* Neural Networks
* Search Engines
* Computer Vision
* Indexing
* Vector Space Models
* Text Mining
* Transformers

The document corpus is described in the project report on page 3.

---

## 2. Text Preprocessing

Before calculating TF-IDF, every document is cleaned using a six-step preprocessing pipeline:

```text
Raw Text
   ↓
Lowercase Conversion
   ↓
Remove Non-Letter Characters
   ↓
Tokenization
   ↓
Stop-Word Removal
   ↓
Remove Single-Character Tokens
   ↓
Porter Stemming
   ↓
Processed Text
```

### Example

```text
Original:
Deep learning architectures for natural language processing

Processed:
deep learn architect natur languag process
```

The same preprocessing pipeline is applied to both documents and user queries.

---

# 📊 TF-IDF

The system converts processed documents into numerical vectors using **TF-IDF**.

### Term Frequency — TF

Measures how frequently a term occurs within a document.

### Inverse Document Frequency — IDF

Measures how rare a term is across the entire corpus.

### TF-IDF

```text
TF-IDF = TF × IDF
```

A term receives a higher weight when it is important within a document but relatively uncommon across the overall corpus.

---

# 🧮 Vector Space Model

Each processed document is represented as a numerical vector.

```text
Document
   ↓
Preprocessing
   ↓
TF-IDF
   ↓
Numerical Vector
```

The resulting TF-IDF matrix represents:

```text
Rows    → Documents
Columns → Vocabulary Terms
Values  → TF-IDF Weights
```

For the project corpus, the system generated:

```text
Documents:       10
Unique Features: 65
Matrix Size:     10 × 65
```

The report notes that the resulting matrix is sparse because most documents do not contain every vocabulary term.

---

# 🔍 Query Processing

When a user enters a query, the query goes through the **same preprocessing pipeline** as the documents.

For example:

```text
User Query:
vector space models and tf-idf weighting

        ↓

Preprocessing

        ↓

vector space model tfidf weight

        ↓

TF-IDF Vector

        ↓

Cosine Similarity

        ↓

Ranked Documents
```

An important implementation detail is that the existing TF-IDF vectorizer is reused with `transform()` rather than fitting a new vectorizer on the query.

---

# 📐 Cosine Similarity

Cosine similarity measures the similarity between the query vector and each document vector.

### Formula

```text
             d · q
cos(d,q) = ---------
            ||d|| ||q||
```

The score ranges from:

```text
0 → No similarity
1 → Maximum similarity
```

The system calculates the similarity score for every document and sorts the results in descending order.

---

# 🏆 Example Search Results

### Query 1

```text
vector space models and tf-idf weighting
```

Example ranking:

| Rank | Document | Similarity |
| ---: | -------- | ---------: |
|    1 | Doc 2    |     0.7414 |
|    2 | Doc 8    |     0.3990 |
|    3 | Doc 10   |     0.1030 |

`Doc 2` receives the highest score because it directly discusses vector space models and TF-IDF weighting.

---

### Query 2

```text
natural language processing
```

Example ranking:

| Rank | Document | Similarity |
| ---: | -------- | ---------: |
|    1 | Doc 1    |     0.5288 |
|    2 | Doc 5    |     0.4930 |
|    3 | Doc 10   |     0.4797 |

The results demonstrate how TF-IDF weighting can rank documents based on the importance and distribution of query terms.

---

# 📈 Evaluation

The project includes an evaluation function to measure the effect of preprocessing on the vocabulary.

| Metric                     | Value |
| -------------------------- | ----: |
| Total Raw Word Count       |   103 |
| Total Processed Word Count |    87 |
| Unique Raw Vocabulary      |    71 |
| TF-IDF Features            |    65 |
| Vocabulary Reduction       | 8.45% |

These measurements are reported in the project's evaluation section.

---

# 🧱 Object-Oriented IRSystem

The complete retrieval pipeline was also organized into a reusable Python class:

```python
class IRSystem:
    ...
```

The class handles:

* Document storage
* Text preprocessing
* Stop-word management
* Stemming
* TF-IDF vectorization
* Query processing
* Cosine similarity
* Document ranking
* Top-K result generation

The vectorizer is fitted once when the `IRSystem` object is created, allowing multiple queries to be processed without repeatedly fitting the model.

---

## 💻 Example Usage

A typical search operation can be performed using:

```python
results = ir_system.search(
    "neural network text classification",
    top_k=2
)

print(results)
```

Example result:

```text
Rank  Doc ID  Score
1     Doc 3   0.4968
2     Doc 1   0.3024
```

The project documentation demonstrates this query using the `IRSystem` class.

---

# 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tfidf-document-search-engine.git
cd tfidf-document-search-engine
```

### 2. Install dependencies

```bash
pip install nltk scikit-learn pandas numpy reportlab
```

### 3. Download NLTK Stop Words

The project automatically downloads the required English stop-word dataset:

```python
nltk.download('stopwords')
```

---

# ▶️ Running the Project

Run the main Python program:

```bash
python ir_project_code.py
```

For the numerical TF-IDF demonstration:

```bash
python tfidf_matrix_numerical.py
```

To generate the project documentation:

```bash
python generate_pdf.py
```

---

# 🧠 Learning Outcomes

This project provided practical experience with:

* Information Retrieval fundamentals
* Text preprocessing
* Tokenization
* Stop-word removal
* Stemming
* TF-IDF
* Vector Space Models
* Cosine similarity
* Sparse matrices
* Query ranking
* Object-oriented Python design
* Pandas DataFrames
* Search-engine style document retrieval

The report specifically highlights the importance of preprocessing order, sparse matrices, cosine similarity, and reusing `transform()` for queries.

---

# 🔮 Future Improvements

Possible extensions to this project include:

* 🌐 Web-based search interface
* 📚 Larger document corpus
* ⚡ Faster indexing for large datasets
* 🔎 Boolean and phrase search
* 🧠 Semantic search using embeddings
* 🤖 Transformer-based retrieval
* 📊 Search-result visualization
* 💾 Database-backed document storage
* 📈 Advanced IR evaluation using Precision, Recall, and F1-score

---

# 📄 Documentation

The complete technical documentation contains:

1. Introduction
2. Report Structure
3. Corpus and Text Preprocessing
4. TF-IDF Feature Matrix Construction
5. Query Processing and Cosine Similarity Ranking
6. Pipeline Evaluation Metrics
7. `IRSystem` Class
8. Learning Outcomes
9. Files Submitted

The complete report is **12 pages** and includes implementation details, formulas, code, output screenshots, evaluation results, and the submitted-file structure.

---

# 👨‍💻 Author

**Shaik Laeeq Ahmed**

B.Tech — Artificial Intelligence & Machine Learning

---

## ⭐ Project Highlights

```text
📚 10 Documents
🔤 6-Step Text Preprocessing
📊 65 TF-IDF Features
🧮 Vector Space Model
📐 Cosine Similarity
🏆 Relevance Ranking
🧱 Object-Oriented IRSystem
📈 Pipeline Evaluation
🐍 Python Implementation
```

---

## 📜 License

This project was developed for **academic and educational purposes** as part of an Information Retrieval course assignment.
