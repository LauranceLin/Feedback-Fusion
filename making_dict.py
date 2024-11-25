# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:52:30 2024

@author: user
"""

# # part 1
# from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# # Example Chinese text data
# documents = [
#     "天使紅蝦非常好吃，肉質鮮甜，讓人回味無窮。",
#     "龍蝦湯稍微鹹了一些，但味道還不錯。",
#     "服務生非常親切，服務很好，讓人感到滿意。",
#     "環境非常舒適，是個適合聚會的好地方。",
#     "這裡的天使紅蝦真是一絕，強烈推薦！"
# ]

# # Custom dictionary for Jieba
# jieba.add_word('天使紅蝦')
# jieba.add_word('龍蝦湯')

# # Tokenize documents using Jieba
# tokenized_docs = [' '.join(jieba.lcut(doc)) for doc in documents]

# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(tokenized_docs)

# # Retrieve feature names and scores
# feature_names = vectorizer.get_feature_names_out()
# scores = tfidf_matrix.toarray()

# # Extract and rank keywords for each document
# for doc_idx, score_vector in enumerate(scores):
#     print(f"Document {doc_idx + 1}:")
#     word_scores = [(feature_names[i], score_vector[i]) for i in range(len(score_vector))]
#     sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
#     for word, score in sorted_words[:5]:  # Top 5 words per document
#         print(f"  {word}: {score:.4f}")
# # part 1

# part 2
import numpy as np
from collections import Counter
from itertools import combinations
import pandas as pd

# Tokenize the text and split into sentences
sentences = [
    jieba.lcut("天使紅蝦非常好吃，肉質鮮甜，讓人回味無窮。"),
    jieba.lcut("龍蝦湯稍微鹹了一些，但味道還不錯。"),
    jieba.lcut("服務生非常親切，服務很好，讓人感到滿意。"),
    jieba.lcut("環境非常舒適，是個適合聚會的好地方。"),
    jieba.lcut("這裡的天使紅蝦真是一絕，強烈推薦！")
]

# Flatten and count word frequencies
all_words = [word for sentence in sentences for word in sentence]
word_counts = Counter(all_words)

# Build co-occurrence matrix
unique_words = list(word_counts.keys())
word_index = {word: i for i, word in enumerate(unique_words)}

co_occurrence_matrix = np.zeros((len(unique_words), len(unique_words)))

for sentence in sentences:
    for word1, word2 in combinations(sentence, 2):
        i, j = word_index[word1], word_index[word2]
        co_occurrence_matrix[i][j] += 1
        co_occurrence_matrix[j][i] += 1

# Convert to DataFrame for readability
co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=unique_words, columns=unique_words)

print("Word Frequencies:")
print(word_counts.most_common(10))  # Top 10 frequent words
print("\nCo-occurrence Matrix:")
print(co_occurrence_df)
# part 2