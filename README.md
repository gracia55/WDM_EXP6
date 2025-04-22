## EX6 Information Retrieval Using Vector Space Model in Python
### DATE:15/04/2025
### AIM: 
To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import log, sqrt

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Download the missing 'punkt_tab' data package

documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

query = input("Enter your query: ")

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]

# Preprocess documents and query
preprocessed_docs = {doc_id: preprocess(doc) for doc_id, doc in documents.items()}
preprocessed_query = preprocess(query)

# Vocabulary
vocab = sorted(set(token for doc in preprocessed_docs.values() for token in doc).union(preprocessed_query))

# Term Frequencies
tf = {doc_id: {term: doc.count(term) for term in vocab} for doc_id, doc in preprocessed_docs.items()}
tf["Q"] = {term: preprocessed_query.count(term) for term in vocab}

# Document Frequencies
df = {term: sum(1 for doc in preprocessed_docs.values() if term in doc) for term in vocab}

# Inverse Document Frequencies
N = len(preprocessed_docs)
idf = {term: round(log(N / df[term]), 4) if df[term] != 0 else 0 for term in vocab}

# Weighted frequencies (TF * IDF)
wf = {doc_id: {term: round(tf[doc_id][term] * idf[term], 4) for term in vocab} for doc_id in tf}

# Vector magnitudes
magnitudes = {doc_id: sqrt(sum(w**2 for w in wf[doc_id].values())) for doc_id in wf}

# Dot product and cosine similarity
dot_products = {doc_id: sum(wf["Q"][term] * wf[doc_id][term] for term in vocab) for doc_id in documents}
cos_sim = {doc_id: round(dot_products[doc_id] / (magnitudes["Q"] * magnitudes[doc_id]), 4)
           if magnitudes["Q"] * magnitudes[doc_id] != 0 else 0 for doc_id in documents}

# Creating table
rows = []
for term in vocab:
    row = {
        "Term": term,
        "DF(t)": df[term],
        "IDF(t)": idf[term],
    }
    # Changing doc_id to match the keys in tf dictionary: "doc1", "doc2", "doc3", "doc4", "Q"
    for doc_id in ["doc1", "doc2", "doc3", "Q"]:
        row[f"TF({doc_id})"] = tf[doc_id][term]
        row[f"WF({doc_id})"] = wf[doc_id][term]
    rows.append(row)

df_table = pd.DataFrame(rows)

# Display core table
print("\nTF/DF/IDF/WF Table:\n")
print(df_table.to_string(index=False))

# Cosine Similarity Table
print("\nVector Magnitudes:")
for doc_id, mag in magnitudes.items():
    print(f"|{doc_id}| = {round(mag, 4)}")

print("\nDot Products (Q · Dᵢ):")
for doc_id, dot in dot_products.items():
    print(f"Q · {doc_id} = {round(dot, 4)}")

print("\nCosine Similarities:")
ranked = sorted(cos_sim.items(), key=lambda x: x[1], reverse=True)
for doc_id, score in ranked:
    print(f"Cosine(Q, {doc_id}) = {score}")

print("\nRanking based on Cosine Similarity:")
for i, (doc_id, score) in enumerate(ranked, 1):
    print(f"Rank {i}: {doc_id}")
```
### Output:
#### Display core table
![Screenshot 2025-04-21 143915](https://github.com/user-attachments/assets/b9759fea-b51d-4941-99f0-cf7176d6bebc)

#### Cosine Similarity Table
![Screenshot 2025-04-21 143923](https://github.com/user-attachments/assets/9e1005bd-85b7-4821-9fd4-01df71e21f8c)

### Result:
Thus the implementation of Information Retrieval Using Vector Space Model in Python is runned successfully.
