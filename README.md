# Information-Retrieval-RA2411026050002-

Information Retrieval (IR) Project
TF-IDF Search Engine with Cosine Similarity Ranking & Interactive Numerical Calculator

📌 Project Overview
This repository contains a complete, scratch-built implementation of an Information Retrieval (IR) System using the Vector Space Model (VSM), TF-IDF weighting, and Cosine Similarity ranking. Developed as part of coursework requirements at SRM Institute of Science and Technology, the project features a fully modular Python pipeline, an Object-Oriented wrapper class (IRSystem), and an interactive step-by-step calculator designed for exam practice and manual verification.  
PDF
+ 1

📂 Repository Structure
Plaintext


├── Information Retrieval/
│   ├── ir_project_code.py          # Main end-to-end IR script (Preprocessing, TF-IDF, Search, Evaluation)
│   ├── tfidf_matrix_numerical.py   # Interactive step-by-step TF-IDF manual calculator tool
│   ├── IR_Project_Documentation.pdf # Comprehensive project report 
│   └── README.md                   # Project documentation
🚀 Key Features & Modules
1. Robust Text Preprocessing Pipeline
To reduce noise and normalize text before mathematical modeling, a strict 6-step cleaning sequence is applied to both the corpus and incoming queries:  
PDF

Lowercase Conversion: Unifies capitalization (e.g., 'Apple' and 'apple').  
PDF

Regex Filtering: Removes non-alphabetic characters.  
PDF

Tokenization: Splits strings on whitespace.  
PDF

Stop-Word Removal: Eliminates standard English stop words using NLTK.  
PDF

Length Filtering: Drops tokens of length 1 or 0.  
PDF

Porter Stemming: Collapses inflected variants into root forms using NLTK PorterStemmer (e.g., classifier, classification → classifi).  
PDF

2. TF-IDF Matrix Construction & Vector Space Model
Built using scikit-learn's TfidfVectorizer.  
PY
+ 1

Computes Term Frequency (TF), Inverse Document Frequency (IDF), and their product (TF-IDF) across a benchmark corpus of research documents.  
PY
+ 1

3. Cosine Similarity Ranking
Computes the angle between normalized query vectors and document vectors to rank relevant search results regardless of document length.  
PDF

Prevents data leakage by utilizing .transform() (instead of .fit_transform()) on user queries.  
PDF

4. Object-Oriented Refactoring (IRSystem Class)
Encapsulates the entire workflow into a reusable class.  
PDF

Instantiates the vectorizer once, allowing multiple high-performance .search() queries without re-fitting.  
PDF

5. Interactive Numerical Calculation Tool (tfidf_matrix_numerical.py)
An interactive command-line utility allowing users to input custom frequency matrices or use default sets.  
PY

Computes step-by-step TF, IDF (ln(N/df)), and final TF-IDF scores with detailed formula breakdowns, ideal for exam preparation.  
PY
+ 1

🛠️ Installation & Requirements
Ensure you have Python installed along with the required libraries:  
PDF

Bash


pip install nltk scikit-learn pandas numpy reportlab
💻 Usage Instructions
Running the Main Search Engine Script
Execute the end-to-end pipeline to view preprocessing outputs, TF-IDF matrix summaries, search rankings, and evaluation metrics:  
PY

Bash


python ir_project_code.py
Running the Interactive TF-IDF Exam Calculator
Launch the interactive tool to test custom frequency matrices or check individual term calculations:  
PY
+ 1

Bash


python tfidf_matrix_numerical.py
📊 Sample Output Preview
Search Query Example:

Plaintext


RAW QUERY       : 'vector space models and tf-idf weighting'
PROCESSED QUERY : 'vector space model tfidf weight'
Rank  Doc ID    Score     Document Snippet
----------------------------------------------------------
1     Doc 2     0.7414    Information retrieval systems utilizing vect...
2     Doc 8     0.3990    Query optimization techniques in classical v...
3     Doc 10    0.1030    Transformer models and self attention mechan...
